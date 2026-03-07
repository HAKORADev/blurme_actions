#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/extensions/shape.h>
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
#include <unistd.h>

class VolumeManager {
    static bool _was_muted;
public:
    static void mute() {
        if (system("pactl set-sink-mute @DEFAULT_SINK@ 1 2>/dev/null") != 0) {
            system("amixer -D pulse sset Master mute 2>/dev/null");
            system("amixer sset Master mute 2>/dev/null");
        }
        _was_muted = true;
    }
    static void unmute() {
        if (!_was_muted) return;
        system("pactl set-sink-mute @DEFAULT_SINK@ 0 2>/dev/null");
        system("amixer -D pulse sset Master unmute 2>/dev/null");
        system("amixer sset Master unmute 2>/dev/null");
        _was_muted = false;
    }
};
bool VolumeManager::_was_muted = false;

struct Config {
    std::string blur_mode = "colored";
    int blurness  = 50;
    int opacity   = 255;
    int grayness  = 128;
    int sound     = 0;
};

class ConfigManager {
public:
    Config      config;
    std::string config_path;

    ConfigManager() {
        char buf[4096] = {};
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            std::string exe(buf, (size_t)n);
            size_t slash = exe.rfind('/');
            config_path = (slash != std::string::npos
            ? exe.substr(0, slash + 1) : "./") + "blur.conf";
        } else {
            config_path = "blur.conf";
        }
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
                if      (key == "blur")     config.blur_mode = value;
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
        f << "blur = "     << config.blur_mode << "\n"
        << "blurness = " << config.blurness  << "\n"
        << "opacity = "  << config.opacity   << "\n"
        << "grayness = " << config.grayness  << "\n"
        << "sound = "    << config.sound     << "\n";
    }
};

static void boxBlurH(const unsigned char* __restrict__ src,
                     unsigned char* __restrict__ dst,
                     int w, int h, int r)
{
    const float iarr = 1.0f / (r + r + 1);
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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
    - n * wl * wl - 4 * n * wl - 3 * n)
    / (-4.0f * wl - 4.0f);
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
    #pragma omp parallel for schedule(static)
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

class BlurMe {
    Display* dpy     = nullptr;
    Window   root    = 0;
    Window   win     = 0;
    int      scr     = 0;
    int      sw      = 0;
    int      sh      = 0;
    int      depth   = 0;
    Visual*  visual  = nullptr;

    ConfigManager cfg;
    int  blur_radius;
    int  opacity;
    bool gray_mode;
    int  gray_level;
    bool mute_sound;

    std::atomic<bool> running      {false};
    std::atomic<bool> enabled      {false};
    std::atomic<bool> first_frame  {false};
    std::atomic<bool> stopped      {false};

    std::thread cap_thread;
    std::thread key_thread;

    XImage* ximg   = nullptr;
    Pixmap  pixmap = 0;
    GC      gc     = nullptr;

    std::vector<unsigned char> frame_buf;

    Atom a_wm_state      = 0;
    Atom a_above         = 0;
    Atom a_fullscreen    = 0;
    Atom a_stay_on_top   = 0;
    Atom a_win_type      = 0;
    Atom a_win_type_dock = 0;
    Atom a_opacity       = 0;

public:
    BlurMe()
    : blur_radius(25), opacity(255), gray_mode(false),
    gray_level(128), mute_sound(true)
    {
        blur_radius = cfg.config.blurness / 2;
        opacity     = cfg.config.opacity;
        gray_mode   = cfg.config.blur_mode == "grayscale";
        gray_level  = cfg.config.grayness;
        mute_sound  = cfg.config.sound == 0;
    }

    ~BlurMe() { stop(); }

    bool init() {
        XInitThreads();
        dpy = XOpenDisplay(nullptr);
        if (!dpy) return false;

        scr    = DefaultScreen(dpy);
        root   = RootWindow(dpy, scr);
        sw     = DisplayWidth(dpy, scr);
        sh     = DisplayHeight(dpy, scr);
        visual = DefaultVisual(dpy, scr);
        depth  = DefaultDepth(dpy, scr);

        internAtoms();
        if (!createWindow()) return false;

        frame_buf.resize(sw * sh * 4);
        running = true;

        cap_thread = std::thread(&BlurMe::captureLoop, this);
        setupKeyGrabs();
        key_thread = std::thread(&BlurMe::keyLoop, this);
        return true;
    }

    void run() {
        while (running)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    void stop() {
        bool exp = false;
        if (!stopped.compare_exchange_strong(exp, true)) return;

        running = false;
        if (enabled && mute_sound) VolumeManager::unmute();

        if (cap_thread.joinable()) cap_thread.join();
        if (key_thread.joinable()) key_thread.join();
        cleanup();
    }

private:
    void internAtoms() {
        a_wm_state      = XInternAtom(dpy, "_NET_WM_STATE",               False);
        a_above         = XInternAtom(dpy, "_NET_WM_STATE_ABOVE",          False);
        a_fullscreen    = XInternAtom(dpy, "_NET_WM_STATE_FULLSCREEN",     False);
        a_stay_on_top   = XInternAtom(dpy, "_NET_WM_STATE_STAYS_ON_TOP",   False);
        a_win_type      = XInternAtom(dpy, "_NET_WM_WINDOW_TYPE",          False);
        a_win_type_dock = XInternAtom(dpy, "_NET_WM_WINDOW_TYPE_DOCK",     False);
        a_opacity       = XInternAtom(dpy, "_NET_WM_WINDOW_OPACITY",       False);
    }

    bool createWindow() {
        XSetWindowAttributes attrs{};
        attrs.override_redirect = True;
        attrs.background_pixel  = 0;
        attrs.border_pixel      = 0;
        attrs.event_mask        = ExposureMask;

        win = XCreateWindow(
            dpy, root, 0, 0, sw, sh, 0, depth,
            InputOutput, visual,
            CWOverrideRedirect | CWBackPixel | CWBorderPixel | CWEventMask,
            &attrs);
        if (!win) return false;

        XChangeProperty(dpy, win, a_win_type, XA_ATOM, 32,
                        PropModeReplace, (unsigned char*)&a_win_type_dock, 1);

        Atom states[] = { a_stay_on_top, a_fullscreen, a_above };
        XChangeProperty(dpy, win, a_wm_state, XA_ATOM, 32,
                        PropModeReplace, (unsigned char*)states, 3);

        Region empty = XCreateRegion();
        XShapeCombineRegion(dpy, win, ShapeInput, 0, 0, empty, ShapeSet);
        XDestroyRegion(empty);

        applyOpacity(opacity);

        gc     = XCreateGC(dpy, win, 0, nullptr);
        pixmap = XCreatePixmap(dpy, win, sw, sh, depth);

        char* buf = (char*)calloc(sw * sh * 4, 1);
        if (!buf) return false;
        ximg = XCreateImage(dpy, visual, depth, ZPixmap, 0,
                            buf, sw, sh, 32, 0);
        return ximg != nullptr;
    }

    void applyOpacity(int op) {
        uint32_t val = (uint32_t)((op / 255.0) * (double)0xFFFFFFFFu);
        XChangeProperty(dpy, win, a_opacity, XA_CARDINAL, 32,
                        PropModeReplace, (unsigned char*)&val, 1);
    }

    void setupKeyGrabs() {
        KeyCode b = XKeysymToKeycode(dpy, XK_b);
        KeyCode c = XKeysymToKeycode(dpy, XK_c);

        unsigned int mods[] = { 0, LockMask, Mod2Mask, LockMask | Mod2Mask };
        for (auto m : mods) {
            XGrabKey(dpy, b, ControlMask | Mod1Mask | m, root, True,
                     GrabModeAsync, GrabModeAsync);
            XGrabKey(dpy, c, ControlMask | Mod1Mask | m, root, True,
                     GrabModeAsync, GrabModeAsync);
            for (KeySym ks : { XK_F1, XK_F2, XK_F3, XK_F4,
                XK_minus,       XK_equal,
                XK_KP_Subtract, XK_KP_Add,
                XK_slash,        XK_asterisk,
                XK_KP_Divide,   XK_KP_Multiply }) {
                XGrabKey(dpy, XKeysymToKeycode(dpy, ks), m, root,
                         True, GrabModeAsync, GrabModeAsync);
                }
        }
        XFlush(dpy);
    }

    void captureLoop() {
        using clock     = std::chrono::steady_clock;
        using us        = std::chrono::microseconds;
        const us frame  = us(1000000 / 30);

        while (running) {
            if (!enabled) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                continue;
            }

            auto t0 = clock::now();
            captureAndDisplay();
            auto elapsed = clock::now() - t0;
            auto sleep   = frame - elapsed;
            std::this_thread::sleep_for(
                sleep > us(1000) ? sleep : us(1000));
        }
    }

    void captureAndDisplay() {
        XImage* img = XGetImage(dpy, root, 0, 0, sw, sh, AllPlanes, ZPixmap);
        if (!img) return;

        memcpy(frame_buf.data(), img->data, (size_t)sw * sh * 4);
        XDestroyImage(img);

        applyGaussianBlur(frame_buf.data(), sw, sh, blur_radius);
        if (gray_mode)
            applyGrayscale(frame_buf.data(), sw, sh, gray_level);

        memcpy(ximg->data, frame_buf.data(), (size_t)sw * sh * 4);
        XPutImage(dpy, pixmap, gc, ximg, 0, 0, 0, 0, sw, sh);
        XCopyArea(dpy, pixmap, win, gc, 0, 0, sw, sh, 0, 0);
        XFlush(dpy);

        if (!first_frame.exchange(true)) {
            XMapRaised(dpy, win);
            XFlush(dpy);
        }
    }

    void keyLoop() {
        XEvent ev;
        while (running) {
            while (XPending(dpy)) {
                XNextEvent(dpy, &ev);
                if (ev.type == KeyPress) {
                    handleKey(XLookupKeysym(&ev.xkey, 0));
                } else if (ev.type == Expose && enabled && pixmap) {
                    XCopyArea(dpy, pixmap, win, gc, 0, 0, sw, sh, 0, 0);
                    XFlush(dpy);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void handleKey(KeySym ks) {
        switch (ks) {
            case XK_b:  toggle();      return;
            case XK_c:  running = false; return;

            case XK_F1:
                gray_mode = false;
                cfg.config.blur_mode = "colored";
                break;
            case XK_F2:
                gray_mode = true;
                cfg.config.blur_mode = "grayscale";
                break;
            case XK_F3:
                gray_level = std::max(0, gray_level - 10);
                cfg.config.grayness = gray_level;
                break;
            case XK_F4:
                gray_level = std::min(255, gray_level + 10);
                cfg.config.grayness = gray_level;
                break;
            case XK_minus: case XK_KP_Subtract:
                opacity = std::max(0, opacity - 4);
                cfg.config.opacity = opacity;
                applyOpacity(opacity);
                break;
            case XK_equal: case XK_plus: case XK_KP_Add:
                opacity = std::min(255, opacity + 4);
                cfg.config.opacity = opacity;
                applyOpacity(opacity);
                break;
            case XK_slash: case XK_KP_Divide:
                blur_radius = std::max(1, blur_radius - 1);
                cfg.config.blurness = blur_radius * 2;
                break;
            case XK_asterisk: case XK_KP_Multiply:
                blur_radius = std::min(100, blur_radius + 1);
                cfg.config.blurness = blur_radius * 2;
                break;
            default: return;
        }
        cfg.save();
    }

    void toggle() {
        enabled = !enabled;
        if (enabled) {
            first_frame = false;
            if (mute_sound) VolumeManager::mute();
        } else {
            if (mute_sound) VolumeManager::unmute();
            XUnmapWindow(dpy, win);
            XFlush(dpy);
        }
    }

    void cleanup() {
        if (ximg) {
            if (ximg->data) { free(ximg->data); ximg->data = nullptr; }
            XDestroyImage(ximg); ximg = nullptr;
        }
        if (pixmap) { XFreePixmap(dpy, pixmap); pixmap = 0; }
        if (gc)     { XFreeGC(dpy, gc);         gc = nullptr; }
        if (win)    { XDestroyWindow(dpy, win);  win = 0; }
        if (dpy)    { XCloseDisplay(dpy);        dpy = nullptr; }
    }
};

int main() {
    BlurMe app;
    if (!app.init()) {
        fprintf(stderr, "BlurMe: failed to open X11 display\n");
        return 1;
    }
    app.run();
    app.stop();
    return 0;
}
