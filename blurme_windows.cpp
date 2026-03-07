#define WIN32_LEAN_AND_MEAN
#define UNICODE
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mutex>

#define WM_BLURME_FRAME    (WM_USER + 1)
#define WM_BLURME_SETTING  (WM_USER + 2)
#define HK_TOGGLE  1
#define HK_CLOSE   2

class VolumeManager {
    static bool _was_muted;
    static void sendMuteToggle() {
        INPUT inp[2] = {};
        inp[0].type       = INPUT_KEYBOARD;
        inp[0].ki.wVk     = VK_VOLUME_MUTE;
        inp[1].type       = INPUT_KEYBOARD;
        inp[1].ki.wVk     = VK_VOLUME_MUTE;
        inp[1].ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(2, inp, sizeof(INPUT));
    }
public:
    static void mute() {
        sendMuteToggle();
        _was_muted = true;
    }
    static void unmute() {
        if (!_was_muted) return;
        sendMuteToggle();
        _was_muted = false;
    }
};
bool VolumeManager::_was_muted = false;

struct Config {
    std::wstring blur_mode = L"colored";
    int blurness = 50;
    int opacity  = 255;
    int grayness = 128;
    int sound    = 0;
};

class ConfigManager {
public:
    Config        config;
    std::wstring  config_path;

    ConfigManager() {
        wchar_t buf[MAX_PATH] = {};
        GetModuleFileNameW(NULL, buf, MAX_PATH);
        std::wstring exe(buf);
        auto slash = exe.rfind(L'\\');
        config_path = (slash != std::wstring::npos
        ? exe.substr(0, slash + 1) : L".\\") + L"blur.conf";
        load();
    }

    void load() {
        std::ifstream f(config_path);
        if (!f.is_open()) { save(); return; }
        std::string line;
        while (std::getline(f, line)) {
            auto pos = line.find('=');
            if (pos == std::string::npos) continue;
            auto trim = [](const std::string& s) -> std::string {
                auto a = s.find_first_not_of(" \t");
                auto b = s.find_last_not_of(" \t");
                return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            std::string key   = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            try {
                if      (key == "blur")     config.blur_mode = std::wstring(value.begin(), value.end());
                else if (key == "blurness") config.blurness  = std::stoi(value);
                else if (key == "opacity")  config.opacity   = std::stoi(value);
                else if (key == "grayness") config.grayness  = std::stoi(value);
                else if (key == "sound")    config.sound     = std::stoi(value);
            } catch (...) {}
        }
    }

    void save() {
        std::ofstream f(config_path);
        if (!f.is_open()) return;
        std::string bm(config.blur_mode.begin(), config.blur_mode.end());
        f << "blur = "     << bm                << "\n"
        << "blurness = " << config.blurness   << "\n"
        << "opacity = "  << config.opacity    << "\n"
        << "grayness = " << config.grayness   << "\n"
        << "sound = "    << config.sound      << "\n";
    }
};

static void boxBlurH(const unsigned char* __restrict__ src,
                     unsigned char* __restrict__ dst,
                     int w, int h, int r)
{
    const float iarr = 1.0f / (r + r + 1);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < h; y++) {
        for (int ch = 0; ch < 3; ch++) {
            int ti = y * w, li = ti, ri = ti + r;
            const int fv = src[ti * 4 + ch];
            const int lv = src[(ti + w - 1) * 4 + ch];
            int val = (r + 1) * fv;
            for (int j = 0; j < r; j++)
                val += src[(ti + j) * 4 + ch];
            for (int j = 0; j <= r; j++) {
                val += src[ri * 4 + ch] - fv;
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                ri++; ti++;
            }
            for (int j = r + 1; j < w - r; j++) {
                val += src[ri * 4 + ch] - src[li * 4 + ch];
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                ri++; li++; ti++;
            }
            for (int j = w - r; j < w; j++) {
                val += lv - src[li * 4 + ch];
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                li++; ti++;
            }
        }
    }
}

static void boxBlurV(const unsigned char* __restrict__ src,
                     unsigned char* __restrict__ dst,
                     int w, int h, int r)
{
    const float iarr = 1.0f / (r + r + 1);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int x = 0; x < w; x++) {
        for (int ch = 0; ch < 3; ch++) {
            int ti = x, li = ti, ri = ti + r * w;
            const int fv = src[ti * 4 + ch];
            const int lv = src[(ti + (h - 1) * w) * 4 + ch];
            int val = (r + 1) * fv;
            for (int j = 0; j < r; j++)
                val += src[(ti + j * w) * 4 + ch];
            for (int j = 0; j <= r; j++) {
                val += src[ri * 4 + ch] - fv;
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                ri += w; ti += w;
            }
            for (int j = r + 1; j < h - r; j++) {
                val += src[ri * 4 + ch] - src[li * 4 + ch];
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                ri += w; li += w; ti += w;
            }
            for (int j = h - r; j < h; j++) {
                val += lv - src[li * 4 + ch];
                dst[ti * 4 + ch] = (unsigned char)(val * iarr + 0.5f);
                li += w; ti += w;
            }
        }
    }
}

static void gaussBoxSizes(float sigma, int n, std::vector<int>& out)
{
    float wf = std::sqrt((12.0f * sigma * sigma / n) + 1.0f);
    int   wl = (int)std::floor(wf);
    if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    float mf = (12.0f * sigma * sigma
    - n * wl * wl - 4 * n * wl - 3 * n) / (-4.0f * wl - 4.0f);
    int m = (int)std::round(mf);
    out.resize(n);
    for (int i = 0; i < n; i++) out[i] = (i < n - m) ? wl : wu;
}

static void applyGaussianBlur(unsigned char* data, int w, int h, int sigma)
{
    if (sigma <= 0) return;
    std::vector<int> bxs;
    gaussBoxSizes((float)sigma, 3, bxs);
    std::vector<unsigned char> tmp(w * h * 4);
    for (int pass = 0; pass < 3; pass++) {
        int r = (bxs[pass] - 1) / 2;
        boxBlurH(data, tmp.data(), w, h, r);
        boxBlurV(tmp.data(), data, w, h, r);
    }
}

static void applyGrayscale(unsigned char* data, int w, int h, int gray_level)
{
    if (gray_level <= 0) return;
    const float alpha = gray_level / 255.0f;
    const int   n     = w * h;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; i++) {
        const float b = data[i * 4];
        const float g = data[i * 4 + 1];
        const float r = data[i * 4 + 2];
        const float L = 0.114f * b + 0.587f * g + 0.299f * r;
        data[i * 4]     = (unsigned char)(b * (1.0f - alpha) + L * alpha);
        data[i * 4 + 1] = (unsigned char)(g * (1.0f - alpha) + L * alpha);
        data[i * 4 + 2] = (unsigned char)(r * (1.0f - alpha) + L * alpha);
    }
}

class BlurMe;
static BlurMe*           g_app     = nullptr;
static LRESULT CALLBACK  WndProc(HWND, UINT, WPARAM, LPARAM);
static LRESULT CALLBACK  LowLevelKeyboardProc(int, WPARAM, LPARAM);

class BlurMe {
    HINSTANCE hInst  = NULL;
    HWND      hwnd   = NULL;

    int vx = 0, vy = 0, vw = 0, vh = 0;

    HDC     dibDC     = NULL;
    HBITMAP dibBmp    = NULL;
    void*   dibPixels = nullptr;

    HDC     capDC  = NULL;
    HBITMAP capBmp = NULL;
    HDC     scrDC  = NULL;

    ConfigManager cfg;
    int  blur_radius;
    int  opacity;
    bool gray_mode;
    int  gray_level;
    bool mute_sound;

    std::atomic<bool> running    {false};
    std::atomic<bool> enabled    {false};
    std::atomic<bool> first_frame{false};
    std::atomic<bool> stopped    {false};

    std::vector<unsigned char> pending_frame;
    std::mutex                 frame_mutex;
    bool                       frame_dirty = false;

    std::thread cap_thread;
    HHOOK       key_hook = NULL;

    static const wchar_t* CLASS_NAME;

public:
    explicit BlurMe(HINSTANCE hi) : hInst(hi) {
        blur_radius = cfg.config.blurness / 2;
        opacity     = cfg.config.opacity;
        gray_mode   = cfg.config.blur_mode == L"grayscale";
        gray_level  = cfg.config.grayness;
        mute_sound  = cfg.config.sound == 0;
    }

    ~BlurMe() { stop(); }

    bool init() {
        vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
        vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
        vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);

        pending_frame.resize((size_t)vw * vh * 4);

        if (!registerWindowClass()) return false;
        if (!createWindow())        return false;
        if (!createGDIResources())  return false;

        running = true;
        cap_thread = std::thread(&BlurMe::captureLoop, this);

        key_hook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, NULL, 0);

        RegisterHotKey(hwnd, HK_TOGGLE, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 'B');
        RegisterHotKey(hwnd, HK_CLOSE,  MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 'C');

        return true;
    }

    void run() {
        MSG msg;
        while (GetMessageW(&msg, NULL, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    void stop() {
        bool exp = false;
        if (!stopped.compare_exchange_strong(exp, true)) return;
        running = false;
        if (enabled && mute_sound) VolumeManager::unmute();
        if (cap_thread.joinable()) cap_thread.join();
        cleanup();
    }

    LRESULT handleMessage(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
        switch (msg) {
            case WM_PAINT:          handlePaint();           return 0;
            case WM_HOTKEY:         handleHotkey((int)wp);   return 0;
            case WM_BLURME_FRAME:   handleFrameReady();      return 0;
            case WM_BLURME_SETTING: handleSettingMsg((DWORD)wp); return 0;
            case WM_DESTROY:        onDestroy();             return 0;
        }
        return DefWindowProcW(h, msg, wp, lp);
    }

    void dispatchHookKey(DWORD vk) {
        if (!hwnd) return;
        PostMessageW(hwnd, WM_BLURME_SETTING, (WPARAM)vk, 0);
    }

private:
    bool registerWindowClass() {
        WNDCLASSEXW wc  = {};
        wc.cbSize       = sizeof(wc);
        wc.lpfnWndProc  = WndProc;
        wc.hInstance    = hInst;
        wc.lpszClassName = CLASS_NAME;
        return RegisterClassExW(&wc) != 0;
    }

    bool createWindow() {
        hwnd = CreateWindowExW(
            WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOPMOST |
            WS_EX_TOOLWINDOW  | WS_EX_NOACTIVATE,
            CLASS_NAME, L"BlurMe",
            WS_POPUP,
            vx, vy, vw, vh,
            NULL, NULL, hInst, NULL);

        if (!hwnd) return false;

        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)this);

        SetLayeredWindowAttributes(hwnd, 0, (BYTE)opacity, LWA_ALPHA);

        SetWindowDisplayAffinity(hwnd, 0x00000011);

        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

        return true;
    }

    bool createGDIResources() {
        BITMAPINFO bmi   = {};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = vw;
        bmi.bmiHeader.biHeight      = -vh;
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        dibBmp = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &dibPixels, NULL, 0);
        if (!dibBmp) return false;
        dibDC = CreateCompatibleDC(NULL);
        SelectObject(dibDC, dibBmp);

        scrDC  = GetDC(NULL);
        capBmp = CreateCompatibleBitmap(scrDC, vw, vh);
        capDC  = CreateCompatibleDC(scrDC);
        SelectObject(capDC, capBmp);

        return true;
    }

    void handlePaint() {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        BitBlt(hdc, 0, 0, vw, vh, dibDC, 0, 0, SRCCOPY);
        EndPaint(hwnd, &ps);
    }

    void handleFrameReady() {
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            if (!frame_dirty) return;
            memcpy(dibPixels, pending_frame.data(), (size_t)vw * vh * 4);
            frame_dirty = false;
        }

        if (!first_frame.exchange(true)) {
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        }

        InvalidateRect(hwnd, NULL, FALSE);
        UpdateWindow(hwnd);
    }

    void handleHotkey(int id) {
        if      (id == HK_TOGGLE) toggle();
        else if (id == HK_CLOSE)  { running = false; DestroyWindow(hwnd); }
    }

    void handleSettingMsg(DWORD vk) {
        switch (vk) {
            case VK_F1:
                gray_mode = false;
                cfg.config.blur_mode = L"colored";
                break;
            case VK_F2:
                gray_mode = true;
                cfg.config.blur_mode = L"grayscale";
                break;
            case VK_F3:
                gray_level = max(0, gray_level - 10);
                cfg.config.grayness = gray_level;
                break;
            case VK_F4:
                gray_level = min(255, gray_level + 10);
                cfg.config.grayness = gray_level;
                break;
            case VK_OEM_MINUS:
            case VK_SUBTRACT:
                opacity = max(0, opacity - 4);
                cfg.config.opacity = opacity;
                SetLayeredWindowAttributes(hwnd, 0, (BYTE)opacity, LWA_ALPHA);
                break;
            case VK_OEM_PLUS:
            case VK_ADD:
                opacity = min(255, opacity + 4);
                cfg.config.opacity = opacity;
                SetLayeredWindowAttributes(hwnd, 0, (BYTE)opacity, LWA_ALPHA);
                break;
            case VK_OEM_2:
            case VK_DIVIDE:
                blur_radius = max(1, blur_radius - 1);
                cfg.config.blurness = blur_radius * 2;
                break;
            case VK_MULTIPLY:
                blur_radius = min(100, blur_radius + 1);
                cfg.config.blurness = blur_radius * 2;
                break;
            default: return;
        }
        cfg.save();
    }

    void onDestroy() {
        stop();
        PostQuitMessage(0);
    }

    void toggle() {
        enabled = !enabled;
        if (enabled) {
            first_frame = false;
            if (mute_sound) VolumeManager::mute();
        } else {
            if (mute_sound) VolumeManager::unmute();
            ShowWindow(hwnd, SW_HIDE);
        }
    }

    void captureLoop() {
        using clock    = std::chrono::steady_clock;
        using us       = std::chrono::microseconds;
        const us frame = us(1000000 / 30);

        BITMAPINFO bmi   = {};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = vw;
        bmi.bmiHeader.biHeight      = -vh;
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        std::vector<unsigned char> local(vw * vh * 4);

        while (running) {
            if (!enabled) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                continue;
            }

            auto t0 = clock::now();

            BitBlt(capDC, 0, 0, vw, vh, scrDC, vx, vy, SRCCOPY | CAPTUREBLT);

            GetDIBits(capDC, capBmp, 0, vh, local.data(), &bmi, DIB_RGB_COLORS);

            applyGaussianBlur(local.data(), vw, vh, blur_radius);
            if (gray_mode)
                applyGrayscale(local.data(), vw, vh, gray_level);

            {
                std::lock_guard<std::mutex> lk(frame_mutex);
                memcpy(pending_frame.data(), local.data(), (size_t)vw * vh * 4);
                frame_dirty = true;
            }
            PostMessageW(hwnd, WM_BLURME_FRAME, 0, 0);

            auto elapsed = clock::now() - t0;
            auto sleep   = frame - elapsed;
            std::this_thread::sleep_for(sleep > us(1000) ? sleep : us(1000));
        }
    }

    void cleanup() {
        if (key_hook) { UnhookWindowsHookEx(key_hook); key_hook = NULL; }
        if (capDC)    { DeleteDC(capDC);       capDC    = NULL; }
        if (capBmp)   { DeleteObject(capBmp);  capBmp   = NULL; }
        if (scrDC)    { ReleaseDC(NULL, scrDC);scrDC    = NULL; }
        if (dibDC)    { DeleteDC(dibDC);       dibDC    = NULL; }
        if (dibBmp)   { DeleteObject(dibBmp);  dibBmp   = NULL; }
        UnregisterHotKey(hwnd, HK_TOGGLE);
        UnregisterHotKey(hwnd, HK_CLOSE);
    }
};

const wchar_t* BlurMe::CLASS_NAME = L"BlurMeOverlay";

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    BlurMe* app = reinterpret_cast<BlurMe*>(
        GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (app) return app->handleMessage(hwnd, msg, wp, lp);
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION &&
        (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) &&
        g_app != nullptr)
    {
        KBDLLHOOKSTRUCT* p = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);
        g_app->dispatchHookKey(p->vkCode);
    }
    return CallNextHookEx(NULL, nCode, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int)
{
    BlurMe app(hInst);
    g_app = &app;

    if (!app.init()) return 1;

    app.run();
    app.stop();
    return 0;
}
