//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------


#include <GL/glew.h>
#include <GL/wglew.h>

#include "main.h"

#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include <windows.h>
#include <windowsx.h>

extern "C" { _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }

HINSTANCE   g_hInstance = 0;
LPSTR       g_lpCmdLine;
int         g_nCmdShow;
bool		g_bARBVerbose = false;

std::vector<NVPWindow *> g_windows;

MSG  uMsg;

struct WINinternal 
{
  NVPWindow * m_win;
  HDC   m_hDC;
  HGLRC m_hRC;
  HWND  m_hWnd;
  HWND  m_hWndDummy;
  bool  m_iconified;
  bool  m_visible;

  WINinternal (NVPWindow *win)
    : m_win(win)
    , m_hDC(NULL)
    , m_hRC(NULL)
    , m_hWnd(NULL)
    , m_hWndDummy(NULL)
    , m_iconified(false)
    , m_visible(true)
  {

  }

  bool create(const char* title, int width, int height);
  bool initBase(const NVPWindow::ContextFlags * cflags, NVPWindow* sourcewindow);
};


//------------------------------------------------------------------------------
// Toggles
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#ifdef _DEBUG
static void APIENTRY myOpenGLCallback(  GLenum source,
                        GLenum type,
                        GLuint id,
                        GLenum severity,
                        GLsizei length,
                        const GLchar* message,
                        const GLvoid* userParam)
{

  NVPWindow* window = (NVPWindow*) userParam;

  GLenum filter = window->m_debugFilter;
  GLenum severitycmp = severity;
  // minor fixup for filtering so notification becomes lowest priority
  if (GL_DEBUG_SEVERITY_NOTIFICATION == filter){
    filter = GL_DEBUG_SEVERITY_LOW_ARB+1;
  }
  if (GL_DEBUG_SEVERITY_NOTIFICATION == severitycmp){
    severitycmp = GL_DEBUG_SEVERITY_LOW_ARB+1;
  }

  if (!filter|| severitycmp <= filter )
  {
  
    //static std::map<GLuint, bool> ignoreMap;
    //if(ignoreMap[id] == true)
    //    return;
    char *strSource = "0";
    char *strType = strSource;
    switch(source)
    {
    case GL_DEBUG_SOURCE_API_ARB:
        strSource = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
        strSource = "WINDOWS";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
        strSource = "SHADER COMP.";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
        strSource = "3RD PARTY";
        break;
    case GL_DEBUG_SOURCE_APPLICATION_ARB:
        strSource = "APP";
        break;
    case GL_DEBUG_SOURCE_OTHER_ARB:
        strSource = "OTHER";
        break;
    }
    switch(type)
    {
    case GL_DEBUG_TYPE_ERROR_ARB:
        strType = "ERROR";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
        strType = "Deprecated";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
        strType = "Undefined";
        break;
    case GL_DEBUG_TYPE_PORTABILITY_ARB:
        strType = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE_ARB:
        strType = "Performance";
        break;
    case GL_DEBUG_TYPE_OTHER_ARB:
        strType = "Other";
        break;
    }
    
	if ( g_bARBVerbose )
		nvprintf ("ARB_DEBUG: %s - %s : %s\n", strSource, strType, message);    
  }
}

//------------------------------------------------------------------------------

#endif

void debugGL ( bool tf )
{
	g_bARBVerbose = tf;
}

void checkGL( const char* msg )
{
	GLenum errCode;	
	errCode = glGetError();
	if (errCode != GL_NO_ERROR) {
		nvprintf ( "%s, GL ERROR: 0x%x\n", msg, errCode );
	}
}


//------------------------------------------------------------------------------
bool WINinternal::initBase(const NVPWindow::ContextFlags* cflags, NVPWindow* sourcewindow)
{
    GLuint PixelFormat;
    
    NVPWindow::ContextFlags  settings;
    if (cflags){
      settings = *cflags;
    }

    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize      = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = settings.depth;
    pfd.cStencilBits = settings.stencil;

    if( settings.stereo )
    {
      pfd.dwFlags |= PFD_STEREO;
    }

    if(settings.MSAA > 1)
    {
        m_hDC = GetDC(m_hWndDummy);
        PixelFormat = ChoosePixelFormat( m_hDC, &pfd );
        SetPixelFormat( m_hDC, PixelFormat, &pfd);
        m_hRC = wglCreateContext( m_hDC );
        wglMakeCurrent( m_hDC, m_hRC );
        glewInit();
        ReleaseDC(m_hWndDummy, m_hDC);
        m_hDC = GetDC( m_hWnd );
        int attri[] = {
			WGL_DRAW_TO_WINDOW_ARB, true,
	        WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
		    WGL_SUPPORT_OPENGL_ARB, true,
			WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
	        WGL_DOUBLE_BUFFER_ARB, true,
	        WGL_DEPTH_BITS_ARB, settings.depth,
	        WGL_STENCIL_BITS_ARB, settings.stencil,
            WGL_SAMPLE_BUFFERS_ARB, 1,
			WGL_SAMPLES_ARB,
			settings.MSAA,
            0,0
        };
        GLuint nfmts;
        int fmt;
	    if(!wglChoosePixelFormatARB( m_hDC, attri, NULL, 1, &fmt, &nfmts )){
            wglDeleteContext(m_hRC);
            return false;
        }
        wglDeleteContext(m_hRC);
        DestroyWindow(m_hWndDummy);
        m_hWndDummy = NULL;
        if(!SetPixelFormat( m_hDC, fmt, &pfd))
            return false;
    } else {
        m_hDC = GetDC( m_hWnd );
        PixelFormat = ChoosePixelFormat( m_hDC, &pfd );
        SetPixelFormat( m_hDC, PixelFormat, &pfd);
    }
    m_hRC = wglCreateContext( m_hDC );
    wglMakeCurrent( m_hDC, m_hRC );

    // calling glewinit NOW because the inside glew, there is mistake to fix...
    // This is the joy of using Core. The query glGetString(GL_EXTENSIONS) is deprecated from the Core profile.
    // You need to use glGetStringi(GL_EXTENSIONS, <index>) instead. Sounds like a "bug" in GLEW.
    glewInit();
#define GLCOMPAT
    if(!wglCreateContextAttribsARB)
        wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
    if(wglCreateContextAttribsARB)
    {
        HGLRC hRC = NULL;
        std::vector<int> attribList;
        #define ADDATTRIB(a,b) { attribList.push_back(a); attribList.push_back(b); }
        int maj= settings.major;
        int min= settings.minor;
        ADDATTRIB(WGL_CONTEXT_MAJOR_VERSION_ARB, maj)
        ADDATTRIB(WGL_CONTEXT_MINOR_VERSION_ARB, min)
        if(settings.core)
            ADDATTRIB(WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB)
        else
            ADDATTRIB(WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB)
        int ctxtflags = 0;
        if(settings.debug)
            ctxtflags |= WGL_CONTEXT_DEBUG_BIT_ARB;
        if(settings.robust)
            ctxtflags |= WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB;
        if(settings.forward) // use it if you want errors when compat options still used
            ctxtflags |= WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;
        ADDATTRIB(WGL_CONTEXT_FLAGS_ARB, ctxtflags);
        ADDATTRIB(0, 0)
        int *p = &(attribList[0]);
        if (!(hRC = wglCreateContextAttribsARB(m_hDC, 0, p )))
        {
            //LOGE("wglCreateContextAttribsARB() failed for OpenGL context.\n");
            return false;
        }
        if (!wglMakeCurrent(m_hDC, hRC)) { 
            //LOGE("wglMakeCurrent() failed for OpenGL context.\n"); 
        } else {
            wglDeleteContext( m_hRC );
            m_hRC = hRC;
#ifdef _DEBUG
            if(!__glewDebugMessageCallbackARB)
            {
                __glewDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC)wglGetProcAddress("glDebugMessageCallbackARB");
                __glewDebugMessageControlARB  = (PFNGLDEBUGMESSAGECONTROLARBPROC) wglGetProcAddress("glDebugMessageControlARB");
            }
            if(__glewDebugMessageCallbackARB)
            {
                glEnable(GL_DEBUG_OUTPUT);
                glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
                glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
                glDebugMessageCallbackARB(myOpenGLCallback, sourcewindow);
            }
#endif
        }
    }
    glewInit();

    //LOGOK("Loaded Glew\n");
    //LOGOK("initialized OpenGL basis\n");
    return true;
}

static int getKeyMods()
{
  int mods = 0;

  if (GetKeyState(VK_SHIFT) & (1 << 31))
    mods |= NVPWindow::KMOD_SHIFT;
  if (GetKeyState(VK_CONTROL) & (1 << 31))
    mods |= NVPWindow::KMOD_CONTROL;
  if (GetKeyState(VK_MENU) & (1 << 31))
    mods |= NVPWindow::KMOD_ALT;
  if ((GetKeyState(VK_LWIN) | GetKeyState(VK_RWIN)) & (1 << 31))
    mods |= NVPWindow::KMOD_SUPER;

  return mods;
}

// Translates a Windows key to the corresponding GLFW key
//
#define INTERNAL_KEY_INVALID -2

static int translateKey(WPARAM wParam, LPARAM lParam)
{
  // Check for numeric keypad keys
  // NOTE: This way we always force "NumLock = ON", which is intentional since
  //       the returned key code should correspond to a physical location.
  if ((HIWORD(lParam) & 0x100) == 0)
  {
    switch (MapVirtualKey(HIWORD(lParam) & 0xFF, 1))
    {
    case VK_INSERT:   return NVPWindow::KEY_KP_0;
    case VK_END:      return NVPWindow::KEY_KP_1;
    case VK_DOWN:     return NVPWindow::KEY_KP_2;
    case VK_NEXT:     return NVPWindow::KEY_KP_3;
    case VK_LEFT:     return NVPWindow::KEY_KP_4;
    case VK_CLEAR:    return NVPWindow::KEY_KP_5;
    case VK_RIGHT:    return NVPWindow::KEY_KP_6;
    case VK_HOME:     return NVPWindow::KEY_KP_7;
    case VK_UP:       return NVPWindow::KEY_KP_8;
    case VK_PRIOR:    return NVPWindow::KEY_KP_9;
    case VK_DIVIDE:   return NVPWindow::KEY_KP_DIVIDE;
    case VK_MULTIPLY: return NVPWindow::KEY_KP_MULTIPLY;
    case VK_SUBTRACT: return NVPWindow::KEY_KP_SUBTRACT;
    case VK_ADD:      return NVPWindow::KEY_KP_ADD;
    case VK_DELETE:   return NVPWindow::KEY_KP_DECIMAL;
    default:          break;
    }
  }

  // Check which key was pressed or released
  switch (wParam)
  {
    // The SHIFT keys require special handling
  case VK_SHIFT:
    {
      // Compare scan code for this key with that of VK_RSHIFT in
      // order to determine which shift key was pressed (left or
      // right)
      const DWORD scancode = MapVirtualKey(VK_RSHIFT, 0);
      if ((DWORD) ((lParam & 0x01ff0000) >> 16) == scancode)
        return NVPWindow::KEY_RIGHT_SHIFT;

      return NVPWindow::KEY_LEFT_SHIFT;
    }

    // The CTRL keys require special handling
  case VK_CONTROL:
    {
      MSG next;
      DWORD time;

      // Is this an extended key (i.e. right key)?
      if (lParam & 0x01000000)
        return NVPWindow::KEY_RIGHT_CONTROL;

      // Here is a trick: "Alt Gr" sends LCTRL, then RALT. We only
      // want the RALT message, so we try to see if the next message
      // is a RALT message. In that case, this is a false LCTRL!
      time = GetMessageTime();

      if (PeekMessage(&next, NULL, 0, 0, PM_NOREMOVE))
      {
        if (next.message == WM_KEYDOWN ||
          next.message == WM_SYSKEYDOWN ||
          next.message == WM_KEYUP ||
          next.message == WM_SYSKEYUP)
        {
          if (next.wParam == VK_MENU &&
            (next.lParam & 0x01000000) &&
            next.time == time)
          {
            // Next message is a RALT down message, which
            // means that this is not a proper LCTRL message
            return INTERNAL_KEY_INVALID;
          }
        }
      }

      return NVPWindow::KEY_LEFT_CONTROL;
    }

    // The ALT keys require special handling
  case VK_MENU:
    {
      // Is this an extended key (i.e. right key)?
      if (lParam & 0x01000000)
        return NVPWindow::KEY_RIGHT_ALT;

      return NVPWindow::KEY_LEFT_ALT;
    }

    // The ENTER keys require special handling
  case VK_RETURN:
    {
      // Is this an extended key (i.e. right key)?
      if (lParam & 0x01000000)
        return NVPWindow::KEY_KP_ENTER;

      return NVPWindow::KEY_ENTER;
    }

    // Funcion keys (non-printable keys)
  case VK_ESCAPE:        return NVPWindow::KEY_ESCAPE;
  case VK_TAB:           return NVPWindow::KEY_TAB;
  case VK_BACK:          return NVPWindow::KEY_BACKSPACE;
  case VK_HOME:          return NVPWindow::KEY_HOME;
  case VK_END:           return NVPWindow::KEY_END;
  case VK_PRIOR:         return NVPWindow::KEY_PAGE_UP;
  case VK_NEXT:          return NVPWindow::KEY_PAGE_DOWN;
  case VK_INSERT:        return NVPWindow::KEY_INSERT;
  case VK_DELETE:        return NVPWindow::KEY_DELETE;
  case VK_LEFT:          return NVPWindow::KEY_LEFT;
  case VK_UP:            return NVPWindow::KEY_UP;
  case VK_RIGHT:         return NVPWindow::KEY_RIGHT;
  case VK_DOWN:          return NVPWindow::KEY_DOWN;
  case VK_F1:            return NVPWindow::KEY_F1;
  case VK_F2:            return NVPWindow::KEY_F2;
  case VK_F3:            return NVPWindow::KEY_F3;
  case VK_F4:            return NVPWindow::KEY_F4;
  case VK_F5:            return NVPWindow::KEY_F5;
  case VK_F6:            return NVPWindow::KEY_F6;
  case VK_F7:            return NVPWindow::KEY_F7;
  case VK_F8:            return NVPWindow::KEY_F8;
  case VK_F9:            return NVPWindow::KEY_F9;
  case VK_F10:           return NVPWindow::KEY_F10;
  case VK_F11:           return NVPWindow::KEY_F11;
  case VK_F12:           return NVPWindow::KEY_F12;
  case VK_F13:           return NVPWindow::KEY_F13;
  case VK_F14:           return NVPWindow::KEY_F14;
  case VK_F15:           return NVPWindow::KEY_F15;
  case VK_F16:           return NVPWindow::KEY_F16;
  case VK_F17:           return NVPWindow::KEY_F17;
  case VK_F18:           return NVPWindow::KEY_F18;
  case VK_F19:           return NVPWindow::KEY_F19;
  case VK_F20:           return NVPWindow::KEY_F20;
  case VK_F21:           return NVPWindow::KEY_F21;
  case VK_F22:           return NVPWindow::KEY_F22;
  case VK_F23:           return NVPWindow::KEY_F23;
  case VK_F24:           return NVPWindow::KEY_F24;
  case VK_NUMLOCK:       return NVPWindow::KEY_NUM_LOCK;
  case VK_CAPITAL:       return NVPWindow::KEY_CAPS_LOCK;
  case VK_SNAPSHOT:      return NVPWindow::KEY_PRINT_SCREEN;
  case VK_SCROLL:        return NVPWindow::KEY_SCROLL_LOCK;
  case VK_PAUSE:         return NVPWindow::KEY_PAUSE;
  case VK_LWIN:          return NVPWindow::KEY_LEFT_SUPER;
  case VK_RWIN:          return NVPWindow::KEY_RIGHT_SUPER;
  case VK_APPS:          return NVPWindow::KEY_MENU;

    // Numeric keypad
  case VK_NUMPAD0:       return NVPWindow::KEY_KP_0;
  case VK_NUMPAD1:       return NVPWindow::KEY_KP_1;
  case VK_NUMPAD2:       return NVPWindow::KEY_KP_2;
  case VK_NUMPAD3:       return NVPWindow::KEY_KP_3;
  case VK_NUMPAD4:       return NVPWindow::KEY_KP_4;
  case VK_NUMPAD5:       return NVPWindow::KEY_KP_5;
  case VK_NUMPAD6:       return NVPWindow::KEY_KP_6;
  case VK_NUMPAD7:       return NVPWindow::KEY_KP_7;
  case VK_NUMPAD8:       return NVPWindow::KEY_KP_8;
  case VK_NUMPAD9:       return NVPWindow::KEY_KP_9;
  case VK_DIVIDE:        return NVPWindow::KEY_KP_DIVIDE;
  case VK_MULTIPLY:      return NVPWindow::KEY_KP_MULTIPLY;
  case VK_SUBTRACT:      return NVPWindow::KEY_KP_SUBTRACT;
  case VK_ADD:           return NVPWindow::KEY_KP_ADD;
  case VK_DECIMAL:       return NVPWindow::KEY_KP_DECIMAL;

    // Printable keys are mapped according to US layout
  case VK_SPACE:         return NVPWindow::KEY_SPACE;
  case 0x30:             return NVPWindow::KEY_0;
  case 0x31:             return NVPWindow::KEY_1;
  case 0x32:             return NVPWindow::KEY_2;
  case 0x33:             return NVPWindow::KEY_3;
  case 0x34:             return NVPWindow::KEY_4;
  case 0x35:             return NVPWindow::KEY_5;
  case 0x36:             return NVPWindow::KEY_6;
  case 0x37:             return NVPWindow::KEY_7;
  case 0x38:             return NVPWindow::KEY_8;
  case 0x39:             return NVPWindow::KEY_9;
  case 0x41:             return NVPWindow::KEY_A;
  case 0x42:             return NVPWindow::KEY_B;
  case 0x43:             return NVPWindow::KEY_C;
  case 0x44:             return NVPWindow::KEY_D;
  case 0x45:             return NVPWindow::KEY_E;
  case 0x46:             return NVPWindow::KEY_F;
  case 0x47:             return NVPWindow::KEY_G;
  case 0x48:             return NVPWindow::KEY_H;
  case 0x49:             return NVPWindow::KEY_I;
  case 0x4A:             return NVPWindow::KEY_J;
  case 0x4B:             return NVPWindow::KEY_K;
  case 0x4C:             return NVPWindow::KEY_L;
  case 0x4D:             return NVPWindow::KEY_M;
  case 0x4E:             return NVPWindow::KEY_N;
  case 0x4F:             return NVPWindow::KEY_O;
  case 0x50:             return NVPWindow::KEY_P;
  case 0x51:             return NVPWindow::KEY_Q;
  case 0x52:             return NVPWindow::KEY_R;
  case 0x53:             return NVPWindow::KEY_S;
  case 0x54:             return NVPWindow::KEY_T;
  case 0x55:             return NVPWindow::KEY_U;
  case 0x56:             return NVPWindow::KEY_V;
  case 0x57:             return NVPWindow::KEY_W;
  case 0x58:             return NVPWindow::KEY_X;
  case 0x59:             return NVPWindow::KEY_Y;
  case 0x5A:             return NVPWindow::KEY_Z;
  case 0xBD:             return NVPWindow::KEY_MINUS;
  case 0xBB:             return NVPWindow::KEY_EQUAL;
  case 0xDB:             return NVPWindow::KEY_LEFT_BRACKET;
  case 0xDD:             return NVPWindow::KEY_RIGHT_BRACKET;
  case 0xDC:             return NVPWindow::KEY_BACKSLASH;
  case 0xBA:             return NVPWindow::KEY_SEMICOLON;
  case 0xDE:             return NVPWindow::KEY_APOSTROPHE;
  case 0xC0:             return NVPWindow::KEY_GRAVE_ACCENT;
  case 0xBC:             return NVPWindow::KEY_COMMA;
  case 0xBE:             return NVPWindow::KEY_PERIOD;
  case 0xBF:             return NVPWindow::KEY_SLASH;
  case 0xDF:             return NVPWindow::KEY_WORLD_1;
  case 0xE2:             return NVPWindow::KEY_WORLD_2;
  default:               break;
  }

  // No matching translation was found
  return NVPWindow::KEY_UNKNOWN;
}

void NVPWindow::setKeyPress ( int key, bool state )
{
	m_keyToggled[key] = (m_keyPressed[key] != state);
	m_keyPressed[key] = state;
}


//------------------------------------------------------------------------------
LRESULT CALLBACK WindowProc( HWND   m_hWnd, 
                             UINT   msg, 
                             WPARAM wParam, 
                             LPARAM lParam )
{
    bool bRes = false;
    // get back the correct window
    int index = (int)GetWindowLongPtr(m_hWnd, GWLP_USERDATA);
    NVPWindow *pWin = NULL;
    if(g_windows.size() > 0)
        pWin = g_windows[index];
    //
    // Pass the messages to our UI, first
    //
    if(!bRes) switch( msg )
    {
        case WM_ACTIVATE:
            if(pWin->m_internal)
                pWin->m_internal->m_iconified = HIWORD(wParam) ? true : false;
            break;
        case WM_SHOWWINDOW:
            if(pWin->m_internal)
                pWin->m_internal->m_visible = wParam ? true : false;
            break;
        case WM_PAINT:
            pWin->postRedisplay();
            break;

        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
          {
                pWin->postRedisplay();

                const int scancode = (lParam >> 16) & 0xff;
                const int key = translateKey(wParam, lParam);
                if (key == INTERNAL_KEY_INVALID)
                  break;

                pWin->setMods(getKeyMods());

				pWin->setKeyPress ( key, true );

                pWin->keyboard( (NVPWindow::KeyCode)key,NVPWindow::BUTTON_PRESS, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
                break;
          }

        case WM_KEYUP:
        case WM_SYSKEYUP:
            {
                pWin->postRedisplay();

                const int scancode = (lParam >> 16) & 0xff;
                const int key = translateKey(wParam, lParam);
                if (key == INTERNAL_KEY_INVALID)
                  break;

                pWin->setMods(getKeyMods());

				pWin->setKeyPress ( key, false );

                pWin->keyboard( (NVPWindow::KeyCode)key,NVPWindow::BUTTON_RELEASE, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
                break;
            }

        case WM_CHAR:
        case WM_SYSCHAR:
            {
                unsigned int key = (unsigned int)wParam;
                if (key < 32 || (key > 126 && key < 160))
                    break;				

				pWin->setKeyPress ( key, true );
                pWin->keyboardchar(key, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            }
            break;
        case WM_MOUSEWHEEL:
            pWin->mousewheel((short)HIWORD(wParam));
            break;
        case WM_LBUTTONDBLCLK:
            pWin->setCurMouse(GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_LEFT, NVPWindow::BUTTON_REPEAT, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_LBUTTONDOWN:			
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam) );			
            pWin->mouse(NVPWindow::MOUSE_BUTTON_LEFT, NVPWindow::BUTTON_PRESS, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_LBUTTONUP:
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_LEFT, NVPWindow::BUTTON_RELEASE, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_RBUTTONDOWN:
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_RIGHT, NVPWindow::BUTTON_PRESS, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_RBUTTONUP:
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_RIGHT, NVPWindow::BUTTON_RELEASE, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_MBUTTONDOWN:
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_MIDDLE, NVPWindow::BUTTON_PRESS, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_MBUTTONUP:
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
            pWin->mouse(NVPWindow::MOUSE_BUTTON_MIDDLE, NVPWindow::BUTTON_RELEASE, pWin->getMods(), pWin->getCurX(), pWin->getCurY());
            break;
        case WM_MOUSEMOVE: {
			int dx = pWin->getCurX();
			int dy = pWin->getCurY();
            pWin->setCurMouse( GET_X_LPARAM(lParam), GET_Y_LPARAM (lParam));
			dx -= (dx==-1) ? 0 : pWin->getCurX();
			dy -= (dy==-1) ? 0 : pWin->getCurY();
            pWin->motion(pWin->getCurX(), pWin->getCurY(), dx, dy);
						} break;
		case WM_SIZE: {			
			int sw = LOWORD(lParam), sh = HIWORD(lParam);
			if (sw > 0 && sh > 0) {
				pWin->setWinSz(sw, sh);
				pWin->reshape(sw, sh);
			}
			} break;
		case WM_SYSCOMMAND:
			switch (wParam) {			
			case SC_MAXIMIZE: pWin->setFullscreen(true);	break;			
			}
			break;
		case WM_QUIT: case WM_CLOSE: 
			// Close window. Deactivate one of the windows.
			pWin->m_active = false;
			break;
		case WM_DESTROY:
			// Proper shutdown. Only WM_DESTROY calls PostQuitMessage
			PostQuitMessage ( 0 );			
            break;
        default:
            break;
    }
    return DefWindowProc( m_hWnd, msg, wParam, lParam );
}
//------------------------------------------------------------------------------
bool WINinternal::create(const char* title, int width, int height)
{                      
  WNDCLASSEX winClass;

  winClass.lpszClassName = "MY_WINDOWS_CLASS";
  winClass.cbSize        = sizeof(WNDCLASSEX);
  winClass.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
  winClass.lpfnWndProc   = WindowProc;
  winClass.hInstance     = g_hInstance;
  winClass.hIcon         = LoadIcon(g_hInstance, IDI_APPLICATION );
  winClass.hIconSm       = LoadIcon(g_hInstance, IDI_APPLICATION );
  winClass.hCursor       = LoadCursor(NULL, IDC_ARROW);
  winClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
  winClass.lpszMenuName  = NULL;
  winClass.cbClsExtra    = 0;
  winClass.cbWndExtra    = 0;

  if(!RegisterClassEx(&winClass) )
    return false;

  DWORD style = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SIZEBOX;
  DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

  RECT rect = { 0, 0, width, height };
  AdjustWindowRectEx(&rect, style,
    FALSE, styleEx);

  m_hWnd = CreateWindowEx( styleEx, "MY_WINDOWS_CLASS",
    title ? title : "Viewer",
    style, 0, 0, rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, 
    g_hInstance, (LPVOID)NULL );
  winClass.lpszClassName = "DUMMY";
  winClass.lpfnWndProc   = DefWindowProc;
  if(!RegisterClassEx(&winClass) )
    return false;
  m_hWndDummy = CreateWindowEx( NULL, "DUMMY",
    "Dummy",
    WS_OVERLAPPEDWINDOW, 0, 0, 10, 10, NULL, NULL, 
    g_hInstance, NULL );

  if( m_hWnd == NULL )
    return false;

  return true;
}

bool NVPWindow::activate(int width, int height, const char* title, const ContextFlags* cflags, int invisible)
{
  // FIXME invisibile handling!
  return create(title,cflags,width,height);
}

void NVPWindow::deactivate()
{
  // FIXME should remove from g_windows
}

static const char *g_screenquad_vert =
	"#version 440 core\n"
	"layout(location = 0) in vec3 vertex;\n"
	"layout(location = 1) in vec3 normal;\n"
	"layout(location = 2) in vec3 texcoord;\n"
	"uniform vec4 uCoords;\n"
	"uniform vec2 uScreen;\n"
	"out vec3 vtc;\n"
	"void main() {\n"
	"   vtc = texcoord*0.5+0.5;\n"
	"   gl_Position = vec4( -1.0 + (uCoords.x/uScreen.x) + (vertex.x+1.0f)*(uCoords.z-uCoords.x)/uScreen.x,\n"
	"                       -1.0 + (uCoords.y/uScreen.y) + (vertex.y+1.0f)*(uCoords.w-uCoords.y)/uScreen.y,\n"
	"                       0.0f, 1.0f );\n"
	"}\n";

static const char *g_screenquad_frag =
	"#version 440\n"
	"uniform sampler2D uTex1;\n"
	"uniform sampler2D uTex2;\n"
	"uniform int uTexFlags;\n"
	"in vec3 vtc;\n"
	"out vec4 outColor;\n"
	"void main() {\n"
	"   vec4 op1 = ((uTexFlags & 0x01)==0) ? texture ( uTex1, vtc.xy) : texture ( uTex1, vec2(vtc.x, 1.0-vtc.y));\n"
	"   if ( (uTexFlags & 0x02) != 0 ) {\n"
	"		vec4 op2 = ((uTexFlags & 0x04)==0) ? texture ( uTex2, vtc.xy) : texture ( uTex2, vec2(vtc.x, 1.0-vtc.y));\n"
	"		outColor = vec4( op1.xyz*(1.0-op2.w) + op2.xyz * op2.w, 1 );\n"
	"   } else { \n"
	"		outColor = vec4( op1.xyz, 1 );\n"
	"   }\n"
	"}\n";


struct nvVertex {
	nvVertex(float x1, float y1, float z1, float tx1, float ty1, float tz1) { x=x1; y=y1; z=z1; tx=tx1; ty=ty1; tz=tz1; }
	float	x, y, z;
	float	nx, ny, nz;
	float	tx, ty, tz;
};
struct nvFace {
	nvFace(unsigned int x1, unsigned int y1, unsigned int z1) { a=x1; b=y1; c=z1; }
	unsigned int  a, b, c;
};

void NVPWindow::initScreenQuadGL()
{
	int status;
	int maxLog = 65536, lenLog;
	char log[65536];

	// Create a screen-space shader
	m_screenquad_prog = (int)glCreateProgram();
	GLuint vShader = (int)glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShader, 1, (const GLchar**)&g_screenquad_vert, NULL);
	glCompileShader(vShader);
	glGetShaderiv(vShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(vShader, maxLog, &lenLog, log);
		nvprintf("*** Compile Error in init_screenquad vShader\n");
		nvprintf("  %s\n", log);
	}

	GLuint fShader = (int)glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShader, 1, (const GLchar**)&g_screenquad_frag, NULL);
	glCompileShader(fShader);
	glGetShaderiv(fShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(fShader, maxLog, &lenLog, log);
		nvprintf("*** Compile Error in init_screenquad fShader\n");
		nvprintf("  %s\n", log);
	}
	glAttachShader(m_screenquad_prog, vShader);
	glAttachShader(m_screenquad_prog, fShader);
	glLinkProgram(m_screenquad_prog);
	glGetProgramiv(m_screenquad_prog, GL_LINK_STATUS, &status);
	if (!status) {
		nvprintf("*** Error! Failed to link in init_screenquad\n");
	}
	checkGL("glLinkProgram (init_screenquad)");

	// Get texture params
	m_screenquad_utex1 = glGetUniformLocation(m_screenquad_prog, "uTex1");
	m_screenquad_utex2 = glGetUniformLocation(m_screenquad_prog, "uTex2");
	m_screenquad_utexflags = glGetUniformLocation(m_screenquad_prog, "uTexFlags");
	m_screenquad_ucoords = glGetUniformLocation(m_screenquad_prog, "uCoords");
	m_screenquad_uscreen = glGetUniformLocation(m_screenquad_prog, "uScreen");


	// Create a screen-space quad VBO
	std::vector<nvVertex> verts;
	std::vector<nvFace> faces;
	verts.push_back(nvVertex(-1, -1, 0, -1, 1, 0));
	verts.push_back(nvVertex(1, -1, 0, 1, 1, 0));
	verts.push_back(nvVertex(1, 1, 0, 1, -1, 0));
	verts.push_back(nvVertex(-1, 1, 0, -1, -1, 0));
	faces.push_back(nvFace(0, 1, 2));
	faces.push_back(nvFace(2, 3, 0));

	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[0]);
	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[1]);
	checkGL("glGenBuffers (init_screenquad)");
	glGenVertexArrays(1, (GLuint*)&m_screenquad_vbo[2]);
	glBindVertexArray(m_screenquad_vbo[2]);
	checkGL("glGenVertexArrays (init_screenquad)");
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(nvVertex), &verts[0].x, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[V] (init_screenquad)");
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);				// pos
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);	// norm
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);	// texcoord
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * 3 * sizeof(int), &faces[0].a, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[F] (init_screenquad)");
	glBindVertexArray(0);
}

void NVPWindow::initGL()
{
	// additional opengl initialization
	//  (primary init of opengl occurs in WINinteral::initBase)
	initScreenQuadGL ();

	glFinish ();
}

void NVPWindow::createScreenQuadGL ( int* glid, int w, int h )
{
	if ( *glid != -1 ) glDeleteTextures ( 1, (GLuint*) glid );
	glGenTextures ( 1, (GLuint*) glid );
	glBindTexture ( GL_TEXTURE_2D, *glid );
	checkGL ( "glBindTexture (createScreenQuadGL)" );
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 );	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);	
	glTexImage2D  ( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	checkGL ( "glTexImage2D (createScreenQuadGL)" );
	glBindTexture ( GL_TEXTURE_2D, 0 );
}

void NVPWindow::clearScreenGL ()
{
	glClearDepth ( 1.0 );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
}


void NVPWindow::renderScreenQuadGL(int glid, char inv1)
{
	renderScreenQuadGL(glid, -1, (float)0, (float)0, (float)getWidth(), (float)getHeight(), inv1);
}

void NVPWindow::compositeScreenQuadGL(int glid1, int glid2, char inv1, char inv2)
{
	renderScreenQuadGL(glid1, glid2, (float)0, (float)0, (float)getWidth(), (float)getHeight(), inv1, inv2);
}

void NVPWindow::renderScreenQuadGL(int glid1, int glid2, float x1, float y1, float x2, float y2, char inv1, char inv2)
{
	// Prepare pipeline
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_FALSE);
	// Select shader	
	glBindVertexArray(m_screenquad_vbo[2]);
	glUseProgram(m_screenquad_prog);
	checkGL("glUseProgram");
	// Select VBO	
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	checkGL("glBindBuffer");
	// Select texture
	glEnable(GL_TEXTURE_2D);
	glProgramUniform4f(m_screenquad_prog, m_screenquad_ucoords, x1, y1, x2, y2);
	glProgramUniform2f(m_screenquad_prog, m_screenquad_uscreen, (float)getWidth(), (float)getHeight());

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, glid1);
	glProgramUniform1i(m_screenquad_prog, m_screenquad_utex1, 0);
	int flags = 0;
	if (inv1 > 0) flags |= 1;												// y-invert tex1

	if (glid2 >= 0) {
		flags |= 2;															// enable tex2 compositing
		if (inv2 > 0) flags |= 4;											// y-invert tex2
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, glid2);
		glProgramUniform1i(m_screenquad_prog, m_screenquad_utex2, 1);
	}
	glProgramUniform1i(m_screenquad_prog, m_screenquad_utexflags, flags);

	// Draw
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
	checkGL("glDraw");
	glUseProgram(0);

	glDepthMask(GL_TRUE);
}


bool NVPWindow::create(const char* title, const ContextFlags* cflags, int width, int height)
{
    m_winSz[0] = width;
    m_winSz[1] = height;
    
    m_internal = new WINinternal(this);
    
    m_debugTitle = title ? title:"Sample";

    if (m_internal->create(m_debugTitle.c_str(), width,height))
    {
      // Keep track of the windows
      g_windows.push_back(this);
      SetWindowLongPtr(m_internal->m_hWnd, GWLP_USERDATA, g_windows.size()-1 );
      UpdateWindow( m_internal->m_hWnd );
      
	  // Initialize the very base of OpenGL
      if (m_internal->initBase(cflags, this) ) {

		// Additional OpenGL initialization
		initGL ();
        
		if( init() ) {
          // showwindow will trigger resize/paint events, that must not be called prior
          // sample init
          ShowWindow( m_internal->m_hWnd, g_nCmdShow );
          return true;
        }
	  }
    }

    

    delete m_internal;
    m_internal = NULL;

    return false;
}

// from file_png.cpp
extern void save_png ( char* fname, unsigned char* img, int w, int h, int ch );

void NVPWindow::save_frame ( char* fname )
{
	int w = getWidth();
	int h = getHeight();

	// Read back pixels
    unsigned char* pixbuf = new unsigned char[w * h * 3];

	glReadPixels ( 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixbuf );

	// Flip Y
	int pitch = w*3;
    unsigned char* buf = new unsigned char[pitch];
	for (int y=0; y < h/2; y++ ) {
		memcpy ( buf, pixbuf + (y*pitch), pitch );		
		memcpy ( pixbuf + (y*pitch), pixbuf + ((h-y-1)*pitch), pitch );		
		memcpy ( pixbuf + ((h-y-1)*pitch), buf, pitch );
	}

	// Save png
	save_png ( fname, pixbuf, w, h, 3 );

    delete[] pixbuf;
    delete[] buf;
}

int NVPWindow::run ( const std::string& title, const std::string& shortname, int argc, const char** argv, int width, int height, int Major, int Minor, int GoldenFrame )
{
    sysVisibleConsole();

	/*
#if _WIN32
	bool m_singleThreaded = true;
    if (m_singleThreaded)
    {
      HANDLE proc = GetCurrentProcess();
      size_t procmask;
      size_t sysmask;
      // pin to one physical cpu for smoother timings, disable hyperthreading
      GetProcessAffinityMask(proc,(PDWORD_PTR)&procmask,(PDWORD_PTR)&sysmask);
      if (sysmask & 8){
        // quadcore, use last core
        procmask = 8;
      }
      else if (sysmask & 2){
        // dualcore, use last core
        procmask = 2;
      }
      SetProcessAffinityMask(proc,(DWORD_PTR)procmask);
    }
#endif*/

    bool vsyncstate = true;
    unsigned int intervalSeconds = 2;
    unsigned int frame = 0;
    unsigned int timerLimit = 0;
    const char* dumpatexit = NULL;

    for (int i = 0; i < argc; i++){
		if (argv[i][0] == '-') {
			on_arg(argv[i], argv[i + 1]);
			i++;
		}		  
		if (strcmp(argv[i],"-vsync") == 0 && i + 1 < argc){
			vsyncstate = atoi(argv[i+1]) ? true : false;
			i+=1;
		}
        if (strcmp(argv[i],"-bmpatexit")==0 && i+1<argc){
          dumpatexit = argv[i+1];
          i++;
        }
    }

    m_cflags.major = Major;
    m_cflags.minor = Minor;

	// Activate
	// - create window
	// - calls app init()
    if (!activate(width,height,title.c_str(), &m_cflags)){
      printf("Could not create GL context: %d.%d\n",m_cflags.major,m_cflags.minor);
      return EXIT_FAILURE;
    }

    vsync( vsyncstate );

    bool Run = begin();
    m_active = true;

    double timeStart = sysGetTime();
    double timeBegin = sysGetTime();    

    bool	lastVsync = m_vsync;
	
	m_display_frame = 0;				// current frame
	m_golden_frame = GoldenFrame;		// golden frame, frame to capture

	char	outpng[1024];				// golden frame, png output filename
	sprintf ( outpng, "out_%s.png", shortname.c_str() );

    if (Run) {
		while ( m_active )
		{
			NVPWindow::sysPollEvents(false);
          
			for(int i=0; i<g_windows.size(); i++)
			{
				NVPWindow *pWin = g_windows[i];
				pWin->m_doSwap = false;
				if ( pWin->m_renderCnt > 0 ) {
	                pWin->m_renderCnt--;				
					pWin->display();
					if ( m_display_frame==m_golden_frame ) save_frame ( outpng );
					m_display_frame++;
					pWin->m_doSwap = true;		
				}
				if ( pWin->m_active == false ) m_active = false;	// Exit the app, if any windows stops.
			}   

			if ( onPress(KEY_V) ) vsync(!m_vsync);
		
			if ( m_doSwap ) 
				swapBuffers();

			if ( m_keyPressed[KEY_ESCAPE] ) {
				m_keyPressed[KEY_ESCAPE] = false;
				if ( m_fullscreen ) {
					restore ();					
				} else {
					// Post close message for proper shutdown
					nvprintf ( "ESC pressed.\n" );
					for(int i=0; i<g_windows.size(); i++)
						PostMessage( g_windows[i]->m_internal->m_hWnd, WM_CLOSE, 0, 0);				
				}
			}
		}
    }
	FreeConsole ();

	m_active = false;

    return Run ? EXIT_SUCCESS : EXIT_FAILURE;
  }

void NVPWindow::vsync (bool state)
{
    swapInterval(state ? 1 : 0);
    m_vsync = state;    
}


void NVPWindow::swapBuffers()
{
    SwapBuffers( m_internal->m_hDC );
}


void NVPWindow::setTitle( const char* title )
{
    SetWindowTextA(m_internal->m_hWnd, title);
}


void NVPWindow::resize_window ( int w, int h )
{	
	RECT rect;
	rect.left = rect.top = 0;
	rect.right = w; rect.bottom = h;
	AdjustWindowRect( &rect, WS_CAPTION, false );
	SetWindowPos ( m_internal->m_hWnd, 0, 0, 0, rect.right-rect.left, rect.bottom-rect.top, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER);
}

void NVPWindow::setFullscreen ( bool fullscreen )
{
  m_fullscreen = fullscreen;

  LONG style = GetWindowLong ( m_internal->m_hWnd, GWL_STYLE );
  LONG style_ex = GetWindowLong ( m_internal->m_hWnd, GWL_EXSTYLE );

  if ( fullscreen ) {

	// Save current size for restore
	m_winSz[2] = m_winSz[0];
	m_winSz[3] = m_winSz[1];

	// Fullscreen	  
    SetWindowLong(m_internal->m_hWnd, GWL_STYLE, style & ~(WS_CAPTION | WS_THICKFRAME));
    SetWindowLong(m_internal->m_hWnd, GWL_EXSTYLE, style_ex & ~(WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE));

    MONITORINFO monitor_info;
    monitor_info.cbSize = sizeof(monitor_info);
    GetMonitorInfo(MonitorFromWindow( m_internal->m_hWnd, MONITOR_DEFAULTTONEAREST), &monitor_info);
    RECT window_rect(monitor_info.rcMonitor);

    SetWindowPos(m_internal->m_hWnd, NULL, window_rect.left, window_rect.top,
                          window_rect.right-window_rect.left, window_rect.bottom-window_rect.top,
                   SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);    
  } else {
    // Restore 
	m_winSz[0] = m_winSz[2];
	m_winSz[1] = m_winSz[3];
		
	SetWindowLong(m_internal->m_hWnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME );
    SetWindowLong(m_internal->m_hWnd, GWL_EXSTYLE, style_ex | WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE );

	SetWindowPos(m_internal->m_hWnd, NULL, 20, 20, m_winSz[0], m_winSz[1], SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
  }
}

void NVPWindow::maximize()
{
	ShowWindow(m_internal->m_hWnd, SW_MAXIMIZE);
}

void NVPWindow::restore()
{
    ShowWindow(m_internal->m_hWnd, SW_RESTORE);
	setFullscreen(false);
}

void NVPWindow::minimize()
{
  ShowWindow(m_internal->m_hWnd, SW_MINIMIZE);
}

bool NVPWindow::isOpen()
{
  return m_internal->m_visible && !m_internal->m_iconified;
}


void NVPWindow::makeContextCurrent()
{
    wglMakeCurrent(m_internal->m_hDC,m_internal->m_hRC);
}

void NVPWindow::makeContextNonCurrent()
{
    wglMakeCurrent(0,0);
}


void NVPWindow::swapInterval(int i)
{
    wglSwapIntervalEXT(i);
}

//---------------------------------------------------------------------------
// Message pump

bool NVPWindow::sysPollEvents(bool bLoop)
{
	while ( PeekMessage( &uMsg, NULL, 0, 0, PM_REMOVE ) )
    { 
        TranslateMessage( &uMsg );
        DispatchMessage( &uMsg );
    }
	return true;
}

// from GLFW 3.0
static int stringInExtensionString(const char* string, const char* exts)
{
  const GLubyte* extensions = (const GLubyte*) exts;
  const GLubyte* start;
  GLubyte* where;
  GLubyte* terminator;

  // It takes a bit of care to be fool-proof about parsing the
  // OpenGL extensions string. Don't be fooled by sub-strings,
  // etc.
  start = extensions;
  for (;;)
  {
    where = (GLubyte*) strstr((const char*) start, string);
    if (!where)
      return GL_FALSE;

    terminator = where + strlen(string);
    if (where == start || *(where - 1) == ' ')
    {
      if (*terminator == ' ' || *terminator == '\0')
        break;
    }

    start = terminator;
  }

  return GL_TRUE;
}

int NVPWindow::sysExtensionSupported( const char* name )
{
  // we are not using the glew query, as glew will only report
  // those extension it knows about, not what the actual driver may support

  int i;
  GLint count;

  // Check if extension is in the modern OpenGL extensions string list
  // This should be safe to use since GL 3.0 is around for a long time :)

  glGetIntegerv(GL_NUM_EXTENSIONS, &count);

  for (i = 0;  i < count;  i++)
  {
    const char* en = (const char*) glGetStringi(GL_EXTENSIONS, i);
    if (!en)
    {
      return GL_FALSE;
    }

    if (strcmp(en, name) == 0)
      return GL_TRUE;
  }

  // Check platform specifc gets

  const char* exts = NULL;
  NVPWindow* win = g_windows[0];

  if (WGLEW_ARB_extensions_string){
    exts = wglGetExtensionsStringARB(win->m_internal->m_hDC);
  }
  if (!exts && WGLEW_EXT_extensions_string){
    exts = wglGetExtensionsStringEXT();
  }
  if (!exts) {
    return FALSE;
  }
  
  return stringInExtensionString(name,exts);
}

NVPWindow::NVPproc NVPWindow::sysGetProcAddress( const char* name )
{
    return (NVPWindow::NVPproc)wglGetProcAddress(name);
}

void NVPWindow::sysWaitEvents()
{
    WaitMessage();
    sysPollEvents(false);
}

static double s_frequency;
double NVPWindow::sysGetTime()
{
    LARGE_INTEGER time;
    if (QueryPerformanceCounter(&time)){
      return (double(time.QuadPart) / s_frequency);
    }
    return 0;
}

void NVPWindow::sysSleep(double seconds)
{
  Sleep(DWORD(seconds * 1000.0));
}

void NVPWindow::sysInit()
{

}

void NVPWindow::sysDeinit()
{

}

static bool s_isConsole = false;
static std::string s_path;

std::string NVPWindow::sysExePath()
{
  return s_path;
}

static const WORD MAX_CONSOLE_LINES = 500;

using namespace std;

void NVPWindow::sysVisibleConsole()
{
  if (s_isConsole) return;

  int hConHandle;
  intptr_t pStdHandle;

  CONSOLE_SCREEN_BUFFER_INFO coninfo;

  FILE *fp;

  // allocate a console for this app
  AllocConsole();

  // set the screen buffer to be big enough to let us scroll text
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),
    &coninfo);

  coninfo.dwSize.Y = MAX_CONSOLE_LINES;
  SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE),
    coninfo.dwSize);

  // redirect unbuffered STDOUT to the console
  pStdHandle = reinterpret_cast<intptr_t>(GetStdHandle(STD_OUTPUT_HANDLE));
  hConHandle = _open_osfhandle(pStdHandle, _O_TEXT);
  fp = _fdopen( hConHandle, "w" );

  *stdout = *fp;

  setvbuf( stdout, NULL, _IONBF, 0 );
  // redirect unbuffered STDIN to the console
  pStdHandle = reinterpret_cast<intptr_t>(GetStdHandle(STD_INPUT_HANDLE));
  hConHandle = _open_osfhandle(pStdHandle, _O_TEXT);
  fp = _fdopen( hConHandle, "r" );

  *stdin = *fp;

  setvbuf( stdin, NULL, _IONBF, 0 );
  // redirect unbuffered STDERR to the console
  pStdHandle = reinterpret_cast<intptr_t>(GetStdHandle(STD_ERROR_HANDLE));
  hConHandle = _open_osfhandle(pStdHandle, _O_TEXT);
  fp = _fdopen( hConHandle, "w" );

  *stderr = *fp;

  setvbuf( stderr, NULL, _IONBF, 0 );

  // make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog

  // point to console as well

  ios::sync_with_stdio();
}

// http://www.codeguru.com/cpp/w-p/win32/article.php/c1427/A-Simple-Win32-CommandLine-Parser.htm
class CmdLineArgs : public std::vector<char*>
{
public:
  CmdLineArgs ()
  {
    // Save local copy of the command line string, because
    // ParseCmdLine() modifies this string while parsing it.
    PSZ cmdline = GetCommandLineA();
    m_cmdline = new char [strlen (cmdline) + 1];
    if (m_cmdline)
    {
      strcpy (m_cmdline, cmdline);
      ParseCmdLine();
    }
  }
  ~CmdLineArgs()
  {
    delete m_cmdline;
  }
private:
  PSZ m_cmdline; // the command line string
  ////////////////////////////////////////////////////////////////////////////////
  // Parse m_cmdline into individual tokens, which are delimited by spaces. If a
  // token begins with a quote, then that token is terminated by the next quote
  // followed immediately by a space or terminator. This allows tokens to contain
  // spaces.
  // This input string: This "is" a ""test"" "of the parsing" alg"o"rithm.
  // Produces these tokens: This, is, a, "test", of the parsing, alg"o"rithm
  ////////////////////////////////////////////////////////////////////////////////
  void ParseCmdLine ()
  {
    enum { TERM = '\0',
          QUOTE = '\"' };
    bool bInQuotes = false;
    PSZ pargs = m_cmdline;
    while (*pargs)
    {
      while (isspace (*pargs)) // skip leading whitespace
        pargs++;
      bInQuotes = (*pargs == QUOTE); // see if this token is quoted
      if (bInQuotes) // skip leading quote
        pargs++;
      push_back (pargs); // store position of current token
      // Find next token.
      // NOTE: Args are normally terminated by whitespace, unless the
      // arg is quoted. That's why we handle the two cases separately,
      // even though they are very similar.
      if (bInQuotes)
      {
        // find next quote followed by a space or terminator
        while (*pargs &&
          !(*pargs == QUOTE && (isspace (pargs[1]) || pargs[1] == TERM)))
          pargs++;
        if (*pargs)
        {
          *pargs = TERM; // terminate token
          if (pargs[1]) // if quoted token not followed by a terminator
            pargs += 2; // advance to next token
        }
      }
      else
      {
        // skip to next non-whitespace character
        while (*pargs && !isspace (*pargs))
          pargs++;
        if (*pargs && isspace (*pargs)) // end of token
        {
          *pargs = TERM; // terminate token
          pargs++; // advance to next token or terminator
        }
      }
    } // while (*pargs)
  } // ParseCmdLine()
}; // class CmdLineArgs


//------------------------------------------------------------------------------

int WINAPI WinMain( HINSTANCE hInstance,
                    HINSTANCE hPrevInstance,
                    LPSTR     lpCmdLine,
                    int       nCmdShow )
{
    g_hInstance = hInstance;
    g_lpCmdLine = lpCmdLine;
    g_nCmdShow = nCmdShow;

    memset(&uMsg,0,sizeof(uMsg));

    LARGE_INTEGER sysfrequency;
    if (QueryPerformanceFrequency(&sysfrequency)){
      s_frequency = (double)sysfrequency.QuadPart;
    }
    else{
      s_frequency = 1;
    }

    CmdLineArgs args;

    std::string exe = args[0];
    std::replace(exe.begin(),exe.end(),'\\','/');

    size_t last = exe.rfind('/');
    if (last != std::string::npos){
      s_path = exe.substr(0,last) + std::string("/");
    }



    //initNSight();
#ifdef MEMORY_LEAKS_CHECK
    _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF); 
    _CrtSetReportMode ( _CRT_ERROR, _CRTDBG_MODE_DEBUG|_CRTDBG_MODE_WNDW);
#endif
    //
    // relay the "main" to the sample
    // the sample will create the window(s)
    //
    sample_main((int)args.size(), (const char**)&args[0]);
    

    //
    // Terminate
    //
    for(int i=0; i<g_windows.size(); i++)
    {
      NVPWindow *pWin = g_windows[i];
      pWin->shutdown();
      if (pWin->m_internal){
        if( pWin->m_internal->m_hRC != NULL )
        {
          ReleaseDC( pWin->m_internal->m_hWnd, pWin->m_internal->m_hDC );
          pWin->m_internal->m_hDC = NULL;
        }
        delete pWin->m_internal;
      }
    }
    UnregisterClass( "MY_WINDOWS_CLASS", g_hInstance );

#ifdef MEMORY_LEAKS_CHECK
    _CrtDumpMemoryLeaks();
#endif
    return (int)uMsg.wParam;
}

int main(int argc, char **argv)
{
  HINSTANCE hinstance = GetModuleHandle(NULL);
  s_isConsole = true;

  WinMain(hinstance, NULL, NULL, 1);
  
  ExitProcess(0);		// also terminates worker threads
  return 0;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
static size_t fmt2_sz    = 0;
static char *fmt2 = NULL;
static FILE *fd = NULL;
static bool bLogReady = false;
static bool bPrintLogging = true;
static int  printLevel = -1; // <0 mean no level prefix

void nvprintSetLevel(int l)
{
    printLevel = l;
}
int nvprintGetLevel()
{
    return printLevel;
}
void nvprintSetLogging(bool b)
{
    bPrintLogging = b;
}
void nvprintf2(va_list &vlist, const char * fmt, int level)
{
    if(bPrintLogging == false)
        return;
    if(fmt2_sz == 0) {
        fmt2_sz = 1024;
        fmt2 = (char*)malloc(fmt2_sz);
    }
    while((vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
    {
        fmt2_sz *= 2;
        if(fmt2) free(fmt2);
        fmt2 = (char*)malloc(fmt2_sz);
    }
#ifdef WIN32
    OutputDebugStringA(fmt2);
#ifdef _DEBUG

    if(bLogReady == false)
    {
        fd = fopen("Log.txt", "w");
        bLogReady = true;
    }
    if(fd)
    {
        //fprintf(fd, prefix);
        fprintf(fd, fmt2);
    }
#endif
#endif
    sample_print(level, fmt2);
    //::printf(prefix);
    ::printf(fmt2);
}
void nvprintf(const char * fmt, ...)
{
//    int r = 0;
    va_list  vlist;
    va_start(vlist, fmt);
    nvprintf2(vlist, fmt, printLevel);
}
void nvprintfLevel(int level, const char * fmt, ...)
{
    va_list  vlist;
    va_start(vlist, fmt);
    nvprintf2(vlist, fmt, level);
}
void nverror ()
{
	nvprintf ( "Error. Application will exit." );	
	exit(-1);
}

bool getFileLocation ( const char* filename, char* outpath )
{
	std::vector<std::string> paths;
	paths.push_back ("./");
	paths.push_back (ASSET_PATH);
	paths.push_back("../shared_assets/");
	bool result = getFileLocation ( filename, outpath, paths );
	return result;
}

bool getFileLocation ( const char* filename, char* outpath, std::vector<std::string> searchPaths )
{
	bool found = false;
	FILE* fp = fopen( filename, "rb" );
	if (fp) {
		found = true;
		strcpy ( outpath, filename );		
	} else {
		for (int i=0; i < searchPaths.size(); i++) {			
			if (searchPaths[i].empty() ) continue;  
			sprintf ( outpath, "%s%s", searchPaths[i].c_str(), filename );
			fp = fopen( outpath, "rb" );
			if (fp)	{ found = true;	break; }
		}		
	}
	if ( found ) fclose ( fp );
	return found;
}
