#include "GymWorker.h"
#include "LawnApp.h"
#include "Resources.h"
#include "GameConstants.h"
#include "Lawn/Board.h"
#include "Lawn/Coin.h"
#include "Lawn/Cutscene.h"
#include "Lawn/Plant.h"
#include "Lawn/LawnMower.h"
#include "Lawn/Projectile.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/Zombie.h"
#include "Lawn/System/PlayerInfo.h"
#include "Lawn/System/Music.h"
#include "Lawn/System/ProfileMgr.h"
#include "Lawn/Widget/SeedChooserScreen.h"
#include "Lawn/Widget/TitleScreen.h"
#include "PvzpLib/PvzpStringFile.h"
#include "PvzpLib/PvzpParticle.h"
#include "PvzpLib/EffectSystem.h"
#include "SexyAppFramework/graphics/GLInterface.h"
#include "SexyAppFramework/misc/MTRand.h"
#include "SexyAppFramework/widget/WidgetManager.h"
#include <SDL.h>
#include <array>
#include <charconv>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
uint32_t Number(const char* text)
{
	uint32_t value;
	const char* end = text + std::strlen(text);
	const auto result = std::from_chars(text, end, value);
	if (result.ec != std::errc() || result.ptr != end)
		throw std::runtime_error("Invalid worker seed or reference flag");
	return value;
}

void WriteWord(uint32_t value)
{
	std::array<unsigned char, 4> bytes;
	for (int i = 0; i < 4; ++i) bytes[i] = (value >> (8 * i)) & 255;
	if (std::fwrite(bytes.data(), 1, bytes.size(), stdout) != bytes.size())
		throw std::runtime_error("Worker response pipe closed");
}

// One process owns all engine globals and one fixed level-1-8 fixture.
class GymApp : public LawnApp {
public:
	void FixedTick()
	{
		mActive = true;
		mMinimized = false;
		if (!DoUpdateFrames()) throw std::runtime_error("Engine did not advance");
		ProcessSafeDeleteList();
	}

	void ScheduledTick()
	{
		SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
		mActive = true;
		mMinimized = false;
		ClearUpdateBacklog(false);
		const auto target = mUpdateCount + 1;
		while (mUpdateCount < target && !mShutdown)
			UpdateAppStep(nullptr);
		if (mUpdateCount != target) throw std::runtime_error("Client scheduler skipped a logical tick");
		ProcessSafeDeleteList();
	}

	void Prepare(uint32_t seed)
	{
		mPlayerInfo = mProfileMgr->AddProfile("Gym");
		if (!mPlayerInfo) throw std::runtime_error("Worker requires a fresh save directory");
		mPlayerInfo->mId = 1;
		mPlayerInfo->mLevel = 8;
		mPlayerInfo->mFinishedAdventure = 0;
		mMusic->MusicTitleScreenInit();
		// Load synchronously so loading-thread timing cannot change RNG consumption.
		LoadingThreadProc();
		if (mLoadingFailed || mShutdown) throw std::runtime_error("Game resources could not be loaded");
		mLoadingThreadCompleted = true;
		mLoaded = true;
		mWidgetManager->RemoveWidget(mTitleScreen.get());
		mTitleScreen.reset();
		mAppRandSeed = seed;
		mRandSeed = seed;
		SRand(seed);
		std::srand(seed);
		mFrameTime = 10;
		mVSyncUpdates = false;
		mUpdateMultiplier = 1;
		mActive = true;
		mMouseIn = true;
		mMinimized = false;
		NewGame();
		// Finish the normal intro with a fixed loadout before episode tick zero.
		const SeedType seeds[] = {SEED_SUNFLOWER, SEED_PEASHOOTER, SEED_CHERRYBOMB,
			SEED_WALLNUT, SEED_POTATOMINE, SEED_SNOWPEA};
		for (int i = 0; !mBoard->mCutScene->mSeedChoosing && i < 2000; ++i) FixedTick();
		if (!mBoard->mCutScene->mSeedChoosing) throw std::runtime_error("Seed chooser did not open");
		for (SeedType seedType : seeds)
			mSeedChooserScreen->ClickedSeedInChooser(mSeedChooserScreen->mChosenSeeds[seedType]);
		for (int i = 0; mSeedChooserScreen->mSeedsInFlight && i < 200; ++i) FixedTick();
		if (mSeedChooserScreen->mSeedsInFlight) throw std::runtime_error("Seed selection did not finish");
		mSeedChooserScreen->OnStartButton();
		for (int i = 0; mGameScene != SCENE_PLAYING && i < 2000; ++i) FixedTick();
		if (mGameScene != SCENE_PLAYING) throw std::runtime_error("Level intro did not finish");
		mWidgetManager->MouseMove(0, 0);
		SDL_SetWindowSize(static_cast<SDL_Window*>(mWindow), BOARD_WIDTH, BOARD_HEIGHT);
		SDL_PumpEvents();
		SDL_GL_SetSwapInterval(0);
		mGLInterface->UpdateViewport();
		if (mGLInterface->mPresentationRect.mWidth != BOARD_WIDTH ||
			mGLInterface->mPresentationRect.mHeight != BOARD_HEIGHT)
			throw std::runtime_error("Worker requires an 800x600 drawable, got " +
				std::to_string(mGLInterface->mPresentationRect.mWidth) + "x" +
				std::to_string(mGLInterface->mPresentationRect.mHeight));
	}

	std::vector<uint8_t> Frame()
	{
		MTAutoDisallowRand disallow;
		glClear(GL_COLOR_BUFFER_BIT);
		mWidgetManager->MarkAllDirty();
		mWidgetManager->DrawScreen();
		return mGLInterface->CaptureRgb();
	}
};

void Mouse(GymApp& app, uint32_t x, uint32_t y, uint32_t event)
{
	if (x >= BOARD_WIDTH || y >= BOARD_HEIGHT || event > 5)
		throw std::runtime_error("Mouse action is outside its declared space");
	if (event == 0) return;
	app.mWidgetManager->MouseMove(x, y);
	if (event == 2 || event == 4) app.mWidgetManager->MouseDown(x, y, event == 2 ? 1 : -1);
	if (event == 3 || event == 5) app.mWidgetManager->MouseUp(x, y, event == 3 ? 1 : -1);
}

// Privileged diagnostics travel only in explicit snapshot responses.
std::string Snapshot(GymApp& app)
{
	std::ostringstream out;
	out << std::setprecision(9) << "{\"scene\":" << app.mGameScene << ",\"rng\":\"";
	constexpr char hex[] = "0123456789abcdef";
	for (unsigned char byte : SerializeRandState()) out << hex[byte >> 4] << hex[byte & 15];
	out << '"';
	Board* b = app.mBoard;
	if (!b) { out << '}'; return out.str(); }
	out << ",\"sun\":" << b->mSunMoney << ",\"sun_countdown\":" << b->mSunCountDown
		<< ",\"wave\":" << b->mCurrentWave << ",\"zombie_countdown\":" << b->mZombieCountDown
		<< ",\"main_counter\":" << b->mMainCounter << ",\"award\":" << b->mLevelAwardSpawned;
	out << ",\"cards\":[";
	for (int i = 0; i < b->mSeedBank->mNumPackets; ++i) {
		auto& p = b->mSeedBank->mSeedPackets[i];
		if (i) out << ',';
		out << '[' << p.mPacketType << ',' << p.mRefreshCounter << ',' << p.mRefreshTime
			<< ',' << p.mRefreshing << ',' << p.mActive << ']';
	}
	out << "],\"plants\":[";
	bool first = true;
	for (Plant* p : b->mPlants) {
		if (!first) out << ','; first = false;
		out << '[' << b->mPlants.DataArrayGetID(p) << ',' << p->mSeedType << ',' << p->mRow
			<< ',' << p->mPlantCol << ',' << p->mPlantHealth << ',' << p->mState << ','
			<< p->mStateCountdown << ',' << p->mLaunchCounter << ',' << p->mDead << ']';
	}
	out << "],\"zombies\":["; first = true;
	for (Zombie* z : b->mZombies) {
		if (!first) out << ','; first = false;
		out << '[' << b->mZombies.DataArrayGetID(z) << ',' << z->mZombieType << ',' << z->mRow
			<< ',' << z->mPosX << ',' << z->mPosY << ',' << z->mZombiePhase << ','
			<< z->mBodyHealth << ',' << z->mHelmHealth << ',' << z->mShieldHealth << ',' << z->mDead << ']';
	}
	out << "],\"projectiles\":["; first = true;
	for (Projectile* p : b->mProjectiles) {
		if (!first) out << ','; first = false;
		out << '[' << b->mProjectiles.DataArrayGetID(p) << ',' << p->mProjectileType << ','
			<< p->mPosX << ',' << p->mPosY << ',' << p->mPosZ << ',' << p->mDead << ']';
	}
	out << "],\"coins\":["; first = true;
	for (Coin* c : b->mCoins) {
		if (!first) out << ','; first = false;
		out << '[' << b->mCoins.DataArrayGetID(c) << ',' << c->mType << ',' << c->mPosX
			<< ',' << c->mPosY << ',' << c->mCoinAge << ',' << c->mDead << ']';
	}
	out << "],\"mowers\":["; first = true;
	for (LawnMower* m : b->mLawnMowers) {
		if (!first) out << ','; first = false;
		out << '[' << b->mLawnMowers.DataArrayGetID(m) << ',' << m->mRow << ',' << m->mPosX
			<< ',' << m->mPosY << ',' << m->mMowerState << ',' << m->mDead << ']';
	}
	out << "],\"particles\":["; first = true;
	auto& particles = app.mEffectSystem->mParticleHolder->mParticles;
	for (PvzpParticle* p : particles) {
		if (!first) out << ','; first = false;
		out << '[' << particles.DataArrayGetID(p) << ',' << p->mParticleAge << ','
			<< p->mPosition.x << ',' << p->mPosition.y << ']';
	}
	out << "]}";
	return out.str();
}

void Respond(GymApp& app, uint32_t tick, int outcome, int reward, bool pixels, bool diagnostic)
{
	const auto frame = pixels ? app.Frame() : std::vector<uint8_t>{};
	const auto state = diagnostic ? Snapshot(app) : std::string{};
	std::fwrite("PVZG", 1, 4, stdout);
	WriteWord(tick); WriteWord(static_cast<uint32_t>(reward)); WriteWord(outcome);
	WriteWord(frame.size()); WriteWord(state.size());
	std::fwrite(frame.data(), 1, frame.size(), stdout);
	std::fwrite(state.data(), 1, state.size(), stdout);
	if (std::fflush(stdout)) throw std::runtime_error("Worker response pipe closed");
}
}

int GymWorkerMain(int argc, char** argv)
{
	try {
		if (argc < 4) throw std::runtime_error("Usage: --gym-worker SEED REFERENCE -resdir=PATH -savedir=PATH");
		const uint32_t seed = Number(argv[2]);
		const uint32_t reference = Number(argv[3]);
		if (reference > 1) throw std::runtime_error("Reference flag must be 0 or 1");
#ifdef _WIN32
		_setmode(_fileno(stdin), _O_BINARY);
		_setmode(_fileno(stdout), _O_BINARY);
#endif
		PvzpStringListSetColors(gLawnStringFormats, gLawnStringFormatCount);
		gExtractResourcesByName = Sexy::ExtractResourcesByName;
		auto app = std::make_unique<GymApp>();
		gLawnApp = app.get();
		std::vector<char*> engineArgs{argv[0]};
		for (int i = 4; i < argc; ++i) engineArgs.push_back(argv[i]);
		app->SetArgs(engineArgs.size(), engineArgs.data());
		app->Init();
		if (app->mShutdown || app->mLoadingFailed) throw std::runtime_error("Engine initialization failed");
		app->SwitchScreenMode(true, app->Is3DAccelerated());
		app->Prepare(seed);
		uint32_t tick = 0;
		int outcome = 0;
		Respond(*app, tick, outcome, 0, true, false);
		std::array<unsigned char, 16> bytes;
		while (true) {
			const size_t n = std::fread(bytes.data(), 1, bytes.size(), stdin);
			if (n == 0 && std::feof(stdin)) break;
			if (n != bytes.size()) throw std::runtime_error("Incomplete worker request");
			std::array<uint32_t, 4> words{};
			for (int w = 0; w < 4; ++w)
				for (int i = 0; i < 4; ++i) words[w] |= uint32_t(bytes[w * 4 + i]) << (8 * i);
			const auto op = words[0];
			if (op == 0) break;
			if (op > 4) throw std::runtime_error("Unknown worker request");
			int reward = 0;
			if (op == 1 || op == 4) {
				if (outcome) throw std::runtime_error("Reset is required after termination");
				Mouse(*app, words[1], words[2], words[3]);
				if (reference) app->ScheduledTick(); else app->FixedTick();
				++tick;
				if (app->mGameScene == SCENE_ZOMBIES_WON) { outcome = 2; reward = -1; }
				else if (!app->mBoard || app->mBoard->mLevelAwardSpawned) { outcome = 1; reward = 1; }
			}
			Respond(*app, tick, outcome, reward, op == 1 || op == 2, op == 3);
		}
		app->Shutdown();
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "Gym worker: %s\n", error.what());
		return 1;
	}
}
