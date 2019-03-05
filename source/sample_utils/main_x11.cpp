//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

#define GLX_GLXEXT_PROTOTYPES

#ifdef _WIN32
  #include<GL/glew.h>
#else
  #include<GL/glew.h>     // DO NOT USE GL/glxew.h (causes segfault)
  #include<GL/glx.h>
  #include<GL/glxext.h>

  #include<X11/Xlib.h>
  #include<X11/Xatom.h>
  #include<X11/keysym.h>
  #include<X11/extensions/xf86vmode.h>
#endif

#include"main.h"
#include<stdio.h>
#include<fcntl.h>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<string>
#include<stdarg.h>
#include<unistd.h>
#include<sys/timeb.h>

Display *g_dpy = 0;

std::vector<NVPWindow*> g_windows;

XEvent uMsg;

typedef GLXContext(*glXCreateContextAttribsARBProc)(Display *,GLXFBConfig, GLXContext,Bool, const int*);

static int attrListDbl[] = {
    GLX_RGBA,GLX_DOUBLEBUFFER,
    GLX_RED_SIZE,4,
    GLX_GREEN_SIZE,4,
    GLX_BLUE_SIZE,4,
    GLX_DEPTH_SIZE,16,0
};

 
static bool ctxErrorOccurred;
static int ctxErrorHandler(Display *dpy, XErrorEvent *evt){
    ctxErrorOccurred = true;
    return 0;
}

//------------------------------------------------------------------------------
// Debug Callback
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

  NVPWindow* window = (NVPWindow*)userParam;

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
    switch(severity)
    {
    case GL_DEBUG_SEVERITY_HIGH_ARB:
        LOGE("ARB_debug : %s High - %s - %s : %s\n", window->m_debugTitle.c_str(), strSource, strType, message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM_ARB:
        LOGW("ARB_debug : %s Medium - %s - %s : %s\n", window->m_debugTitle.c_str(), strSource, strType, message);
        break;
    case GL_DEBUG_SEVERITY_LOW_ARB:
        LOGI("ARB_debug : %s Low - %s - %s : %s\n", window->m_debugTitle.c_str(), strSource, strType, message);
        break;
    default:
        //LOGI("ARB_debug : comment - %s - %s : %s\n", strSource, strType, message);
        break;
    }
  }
}

//------------------------------------------------------------------------------
void checkGL( const char* msg )
{
    GLenum errCode;
    //const GLubyte* errString;
    errCode = glGetError();
    if (errCode != GL_NO_ERROR) {
        //printf ( "%s, ERROR: %s\n", msg, gluErrorString(errCode) );
        LOGE("%s, ERROR: 0x%x\n", msg, errCode );
    }
}

#else
//------------------------------------------------------------------------------
void checkGL( const char* msg ) {}
#endif

struct WINinternal{
    NVPWindow *m_win;
    int m_screen;
    GLXContext m_glx_context;
    GLXFBConfig m_glx_fb_config;
    Display *m_dpy;
    Window m_window;
    XVisualInfo *m_visual;
    XF86VidModeModeInfo m_mode;
    XSetWindowAttributes winAttributes;
    bool m_visible;

    WINinternal(NVPWindow *win):
    m_win(win),
    m_screen(0),
    m_glx_context(0),
    m_glx_fb_config(0),
    m_dpy(0),
    m_window(0),
    m_visual(0),
    m_visible(true)
{}

bool create(const char *title, int width, int height);
bool initBase(const NVPWindow::ContextFlags *cflags, NVPWindow *sourceWindow);

};

bool WINinternal::initBase(const NVPWindow::ContextFlags *cflags, NVPWindow *sourceWindow){

    NVPWindow::ContextFlags settings;
    if(cflags){
        settings = *cflags;
    }

    const char *glxExts = glXQueryExtensionsString(m_dpy,DefaultScreen(m_dpy));

    glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
    glXCreateContextAttribsARB  = (glXCreateContextAttribsARBProc) glXGetProcAddressARB((const GLubyte *) "glXCreateContextAttribsARB");

 
    ctxErrorOccurred = false;
    int (*oldHandler)(Display *, XErrorEvent*) = XSetErrorHandler(&ctxErrorHandler);

    int contextattribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, settings.major,
        GLX_CONTEXT_MINOR_VERSION_ARB, settings.minor,
        GLX_CONTEXT_PROFILE_MASK_ARB, settings.core?GLX_CONTEXT_CORE_PROFILE_BIT_ARB:GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
        0
    }; 

   
    m_glx_context = glXCreateContextAttribsARB(m_dpy,m_glx_fb_config,0,True,contextattribs);


    XSetErrorHandler(oldHandler);

    if(!glXMakeCurrent(m_dpy,m_window,m_glx_context)){
        printf("Error making glx context current.\n");
    }

   GLenum glewErr = glewInit();

   if(GLEW_OK != glewErr){
    printf("Error initialising glew: %s.\n",glewGetErrorString(glewErr));
   }


   return true;
}

static int getKeyMods(XEvent &evt){

    int mods = 0;

    if(evt.xkey.state & ShiftMask){
        mods |= NVPWindow::KMOD_SHIFT;
    }
    if(evt.xkey.state & ControlMask ){
        mods |= NVPWindow::KMOD_CONTROL;
    }
    if(evt.xkey.state & Mod1Mask){
        mods |= NVPWindow::KMOD_ALT;
    }

    return mods;
}

static int translateKey(XEvent &evt, bool& printableKey)
{
    printableKey = false;
    if(evt.type != KeyPress && evt.type != KeyRelease) return 0;

    unsigned int key = evt.xkey.keycode;

    KeySym ksym = XLookupKeysym(&evt.xkey,0);

    switch(ksym){
   //  XLib has separate key symbols for left and right  verions of mod keys. 
        case XK_Shift_L:      return NVPWindow::KEY_LEFT_SHIFT;
        case XK_Shift_R:      return NVPWindow::KEY_RIGHT_SHIFT;
        case XK_Control_L:    return NVPWindow::KEY_LEFT_CONTROL;
        case XK_Control_R:    return NVPWindow::KEY_RIGHT_CONTROL;
        case XK_Alt_L:        return NVPWindow::KEY_LEFT_ALT;
        case XK_Alt_R:        return NVPWindow::KEY_RIGHT_ALT;
        case XK_Return:       return NVPWindow::KEY_ENTER;
        case XK_Escape:       return NVPWindow::KEY_ESCAPE;
        case XK_Tab:          return NVPWindow::KEY_TAB;
        case XK_BackSpace:    return NVPWindow::KEY_BACKSPACE;
        case XK_Home:         return NVPWindow::KEY_HOME;
        case XK_End:          return NVPWindow::KEY_END;
        case XK_Prior:        return NVPWindow::KEY_PAGE_UP;
        case XK_Next:         return NVPWindow::KEY_PAGE_DOWN;
        case XK_Insert:       return NVPWindow::KEY_INSERT;
        case XK_Delete:       return NVPWindow::KEY_DELETE;
        case XK_Left:         return NVPWindow::KEY_LEFT;
        case XK_Right:        return NVPWindow::KEY_RIGHT;
        case XK_Up:           return NVPWindow::KEY_UP;
        case XK_Down:         return NVPWindow::KEY_DOWN;
        case XK_F1:           return NVPWindow::KEY_F1;
        case XK_F2:           return NVPWindow::KEY_F2;
        case XK_F3:           return NVPWindow::KEY_F3;
        case XK_F4:           return NVPWindow::KEY_F4;
        case XK_F5:           return NVPWindow::KEY_F5;
        case XK_F6:           return NVPWindow::KEY_F6;
        case XK_F7:           return NVPWindow::KEY_F7;
        case XK_F8:           return NVPWindow::KEY_F8;
        case XK_F9:           return NVPWindow::KEY_F9;
        case XK_F10:          return NVPWindow::KEY_F10;
        case XK_F11:          return NVPWindow::KEY_F11;
        case XK_F12:          return NVPWindow::KEY_F12;
        case XK_F13:          return NVPWindow::KEY_F13;
        case XK_F14:          return NVPWindow::KEY_F14;
        case XK_F15:          return NVPWindow::KEY_F15;
        case XK_F16:          return NVPWindow::KEY_F16;
        case XK_F17:          return NVPWindow::KEY_F17;
        case XK_F18:          return NVPWindow::KEY_F18;
        case XK_F19:          return NVPWindow::KEY_F19;
        case XK_F20:          return NVPWindow::KEY_F20;
        case XK_Num_Lock:     return NVPWindow::KEY_NUM_LOCK;
        case XK_Caps_Lock:    return NVPWindow::KEY_CAPS_LOCK;
        case XK_Scroll_Lock:  return NVPWindow::KEY_SCROLL_LOCK;
        case XK_Pause:        return NVPWindow::KEY_PAUSE;
        // Numeric Keypad
        case XK_KP_0:         return NVPWindow::KEY_KP_0;
        case XK_KP_1:         return NVPWindow::KEY_KP_1;
        case XK_KP_2:         return NVPWindow::KEY_KP_2;
        case XK_KP_3:         return NVPWindow::KEY_KP_3;
        case XK_KP_4:         return NVPWindow::KEY_KP_4;
        case XK_KP_5:         return NVPWindow::KEY_KP_5;
        case XK_KP_6:         return NVPWindow::KEY_KP_6;
        case XK_KP_7:         return NVPWindow::KEY_KP_7;
        case XK_KP_8:         return NVPWindow::KEY_KP_8;
        case XK_KP_9:         return NVPWindow::KEY_KP_9;
        case XK_KP_Divide:    return NVPWindow::KEY_KP_DIVIDE;
        case XK_KP_Multiply:  return NVPWindow::KEY_KP_MULTIPLY;
        case XK_KP_Subtract:  return NVPWindow::KEY_KP_SUBTRACT;
        case XK_KP_Add:       return NVPWindow::KEY_KP_ADD;
        case XK_KP_Decimal:   return NVPWindow::KEY_KP_DECIMAL;
        case XK_space:        return NVPWindow::KEY_SPACE;

        default:
        break;
    }
//
    // Now processing printable keys
    //
    printableKey = true;
    switch(ksym)
    {
        case XK_0:            return NVPWindow::KEY_0;
        case XK_1:            return NVPWindow::KEY_1;
        case XK_2:            return NVPWindow::KEY_2;
        case XK_3:            return NVPWindow::KEY_3;
        case XK_4:            return NVPWindow::KEY_4;
        case XK_5:            return NVPWindow::KEY_5;
        case XK_6:            return NVPWindow::KEY_6;
        case XK_7:            return NVPWindow::KEY_7;
        case XK_8:            return NVPWindow::KEY_8;
        case XK_9:            return NVPWindow::KEY_9;
        case XK_a:            return 'a';//NVPWindow::KEY_A;
        case XK_b:            return 'b';//NVPWindow::KEY_B;
        case XK_c:            return 'c';//NVPWindow::KEY_C;
        case XK_d:            return 'd';//NVPWindow::KEY_D;
        case XK_e:            return 'e';//NVPWindow::KEY_E;
        case XK_f:            return 'f';//NVPWindow::KEY_F;
        case XK_g:            return 'g';//NVPWindow::KEY_G;
        case XK_h:            return 'h';//NVPWindow::KEY_H;
        case XK_i:            return 'i';//NVPWindow::KEY_I;
        case XK_j:            return 'j';//NVPWindow::KEY_J;
        case XK_k:            return 'k';//NVPWindow::KEY_K;
        case XK_l:            return 'l';//NVPWindow::KEY_L;
        case XK_m:            return 'm';//NVPWindow::KEY_M;
        case XK_n:            return 'n';//NVPWindow::KEY_N;
        case XK_o:            return 'o';//NVPWindow::KEY_O;
        case XK_p:            return 'p';//NVPWindow::KEY_P;
        case XK_q:            return 'q';//NVPWindow::KEY_Q;
        case XK_r:            return 'r';//NVPWindow::KEY_R;
        case XK_s:            return 's';//NVPWindow::KEY_S;
        case XK_t:            return 't';//NVPWindow::KEY_T;
        case XK_u:            return 'u';//NVPWindow::KEY_U;
        case XK_v:            return 'v';//NVPWindow::KEY_V;
        case XK_w:            return 'w';//NVPWindow::KEY_W;
        case XK_x:            return 'x';//NVPWindow::KEY_X;
        case XK_y:            return 'y';//NVPWindow::KEY_Y;
        case XK_z:            return 'z';//NVPWindow::KEY_Z;
        case XK_minus:        return NVPWindow::KEY_MINUS;
        case XK_equal:        return NVPWindow::KEY_EQUAL;
        case XK_bracketleft:  return NVPWindow::KEY_LEFT_BRACKET;
        case XK_bracketright: return NVPWindow::KEY_RIGHT_BRACKET;
        case XK_backslash:    return NVPWindow::KEY_BACKSLASH;
        case XK_semicolon:    return NVPWindow::KEY_SEMICOLON;
        case XK_comma:        return NVPWindow::KEY_COMMA;
        case XK_period:       return NVPWindow::KEY_PERIOD;
    }
    return NVPWindow::KEY_UNKNOWN;
}

bool NVPWindow::activate(int width, int height, const char *title, const ContextFlags *cflags, int invisible){


    return create(title,cflags,width,height);
}

void NVPWindow::deactivate(){

}

bool NVPWindow::create(const char *title, const ContextFlags *cflags, int width, int height){

    m_winSz[0] = width;
    m_winSz[1] = height;

    m_internal = new WINinternal(this);

    m_debugTitle = title ? title:"Sample";

    if(m_internal->create(m_debugTitle.c_str(),width,height)){

        g_windows.push_back(this);

        //  Update and draw.

        if(m_internal->initBase(cflags,this)){
					  
            initGL();
            if(init()){         
     	        return true;     // show window
            }else{
                printf("Init Failed.\n");
            }
        }


    }

    delete m_internal;
    m_internal = NULL;
    return false;
}

int NVPWindow::run ( const std::string& title, const std::string& shortname, int argc, const char** argv, int width, int height, int Major, int Minor, int GoldenFrame )
{
    bool vsyncstate = true;
    unsigned int intervalSeconds = 2;
    unsigned int frameLimit = 0;
    unsigned int timerLimit = 0;
    const char* dumpatexit = NULL;

    for (int i = 0; i < argc; i++){
      if (strcmp(argv[i],"-winsize") == 0 && i + 2 < argc){
        width  = atoi(argv[i+1]);
        height = atoi(argv[i+2]);
        i+=2;
      }
      if (strcmp(argv[i],"-vsync") == 0 && i + 1 < argc){
        vsyncstate = atoi(argv[i+1]) ? true : false;
        i+=1;
      }
      if (strcmp(argv[i],"-frames")==0 && i+1<argc){
        frameLimit = atoi(argv[i+1]);
        i++;
      }
      if (strcmp(argv[i],"-timerprints")==0 && i+1<argc){
        timerLimit = atoi(argv[i+1]);
        i++;
      }
      if (strcmp(argv[i],"-timerinterval")==0 && i+1<argc){
        intervalSeconds = atoi(argv[i+1]);
        i++;
      }
      if (strcmp(argv[i],"-bmpatexit")==0 && i+1<argc){
        dumpatexit = argv[i+1];
        i++;
      }
    }

    m_cflags.major = Major;
    m_cflags.minor = Minor;

    // this calls NVPWindow::create
    if (!activate(width,height,title.c_str(), &m_cflags)){
      printf("Could not create GL context: %d.%d\n",m_cflags.major,m_cflags.minor);
      return EXIT_FAILURE;
    }

    vsync( vsyncstate );

    bool Run = begin();
    m_active = true;

    double timeStart = sysGetTime();
    double timeBegin = sysGetTime();
    double frames = 0;

    bool   lastVsync = m_vsync;

    m_display_frame = 0;
    m_golden_frame = GoldenFrame;
   
  char	outpng[1024];				// golden frame, png output filename
  sprintf ( outpng, "out_%s.png", shortname.c_str() );

    if (Run) {
		while ( m_active )
		{

			NVPWindow::sysPollEvents(false);		// performs display()

			for(uint32_t i=0;i<g_windows.size();++i) {
			   NVPWindow *pWin = g_windows[i];
			   if(pWin->m_renderCnt > 0) {
			      pWin->m_renderCnt--;
			      if ( m_display_frame==m_golden_frame ) save_frame ( outpng );
			      m_display_frame++;
			      pWin->display();
			      pWin->m_doSwap = true;
			   }
			} 
          
			if ( onPress(KEY_V) ) vsync(!m_vsync);
		
			if ( m_doSwap ) 
				swapBuffers();

			frames++;        

			if ( m_keyPressed[KEY_ESCAPE] ) {
			  m_keyPressed[KEY_ESCAPE] = false;
			  if ( m_fullscreen ) {
				restore ();					
			  } else {
			       nvprintf ( "ESC pressed.\n" );
			       m_active = false;	 
                          }
			}
		}
    }

    m_active = false;

    return Run ? EXIT_SUCCESS : EXIT_FAILURE;
  }

void NVPWindow::vsync (bool state)
{
    swapInterval(state ? 1 : 0);
    m_vsync = state;    
}


void NVPWindow::postQuit(){
    //handle X11 quit.
}

void NVPWindow::swapBuffers(){
    WINinternal *win = m_internal;
    glXSwapBuffers(win->m_dpy,win->m_window);
}

void NVPWindow::setTitle(const char *title){

}

void NVPWindow::resize_window ( int w, int h )
{	
    std::cout<<"Attempting to resize window to "<<w<<"x"<<h<<"...\n";
    int result = XResizeWindow(m_internal->m_dpy, m_internal->m_window, w, h);
    std::cout<<result;
}

void NVPWindow::maximize(){
    //handle maximize
}

void NVPWindow::restore(){
    //handle restore
}

void NVPWindow::minimize(){
    //handle minimize
}

bool NVPWindow::isOpen(){
    return m_internal->m_visible;
}

void NVPWindow::makeContextCurrent(){
    glXMakeCurrent(m_internal->m_dpy,m_internal->m_window,m_internal->m_glx_context);
}

void NVPWindow::makeContextNonCurrent(){
    glXMakeCurrent(m_internal->m_dpy,None,NULL);
}

void NVPWindow::swapInterval(int i){
    //do nothing.
  //    glXSwapIntervalEXT (m_internal->m_dpy,m_internal->m_window,i);
}


bool NVPWindow::sysPollEvents(bool bLoop)
{
    NVPWindow::MouseButton btn;   
    bool done = false;
    XEvent event;
    static short mouseWheelScale = 5;

    for(uint32_t i=0; i < g_windows.size(); ++i) {
        NVPWindow *pWin = g_windows[i];

        while(XPending(pWin->m_internal->m_dpy) > 0) {

            XNextEvent(pWin->m_internal->m_dpy,&event);

            switch(event.type){
                case Expose:            break;
                case GraphicsExpose:    break;
                case ConfigureNotify:
                    pWin->setWinSz(event.xconfigure.width, event.xconfigure.height);
                    break;
                case ButtonPress:{
                    pWin->setCurMouse(event.xbutton.x,event.xbutton.y);
                    switch ( event.xbutton.button ) {
                    case Button1: btn = NVPWindow::MOUSE_BUTTON_LEFT; break;
		    case Button2: btn = NVPWindow::MOUSE_BUTTON_MIDDLE; break;
		    case Button3: btn = NVPWindow::MOUSE_BUTTON_RIGHT; break;
		    case Button4: pWin->mousewheel(-mouseWheelScale); break;
		    case Button5: pWin->mousewheel(mouseWheelScale); break;
		    };
                    pWin->mouse(btn, NVPWindow::BUTTON_PRESS, pWin->getMods(),pWin->getCurX(),pWin->getCurY());
                    } break;
                case ButtonRelease:{
                    pWin->setCurMouse(event.xbutton.x,event.xbutton.y);
                    switch ( event.xbutton.button ) {
                    case Button1: btn = NVPWindow::MOUSE_BUTTON_LEFT; break;
		    case Button2: btn = NVPWindow::MOUSE_BUTTON_MIDDLE; break;
		    case Button3: btn = NVPWindow::MOUSE_BUTTON_RIGHT; break;
		    case Button4: pWin->mousewheel(-mouseWheelScale); break;
		    case Button5: pWin->mousewheel(mouseWheelScale); break;
		    };
                    pWin->mouse(btn, NVPWindow::BUTTON_RELEASE, pWin->getMods(),pWin->getCurX(),pWin->getCurY());
                    } break;
 	       case MotionNotify: {
 		    float dx = pWin->getCurX() - event.xmotion.x;
		    float dy = pWin->getCurY() - event.xmotion.y;
                    pWin->setCurMouse(event.xmotion.x,event.xmotion.y);
                    pWin->motion( pWin->getCurX(), pWin->getCurY(), dx, dy );
	            } break;
               case KeyPress:{
		    bool printableKey;
		    int translatedKeyCode = translateKey(event, printableKey);
                    if(translatedKeyCode == NVPWindow::KEY_UNKNOWN) break;
                    pWin->setMods(getKeyMods(event));
                    pWin->m_keyPressed[ translatedKeyCode ] = true;
                    if(printableKey) 
                        pWin->keyboardchar((NVPWindow::KeyCode) translatedKeyCode, pWin->getMods(),pWin->getCurX(),pWin->getCurY());
                    else 
                        pWin->keyboard((NVPWindow::KeyCode) translatedKeyCode, NVPWindow::BUTTON_PRESS,pWin->getMods(),pWin->getCurX(),pWin->getCurY());                                 
                    } break;
                case KeyRelease:{
 		    bool printableKey;
                    int translatedKeyCode = translateKey(event, printableKey);
                    if(translatedKeyCode == NVPWindow::KEY_UNKNOWN) break;
                    pWin->setMods(getKeyMods(event));
                    pWin->m_keyPressed[ translatedKeyCode ] = false;
                    pWin->keyboard((NVPWindow::KeyCode) translatedKeyCode, NVPWindow::BUTTON_RELEASE,pWin->getMods(),pWin->getCurX(),pWin->getCurY());
		    } break;
                case ClientMessage:
                    if(strcmp(XGetAtomName(pWin->m_internal->m_dpy,event.xclient.message_type),"WM_PROTOCOLS") == 0){
                        pWin->shutdown();
                        done = True;
                    }
                 break;            
            }
        }
        XSync(pWin->m_internal->m_dpy, True);
   }

   return true;
}

static int stringInExtensionString(const char *string, const char *exts){
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

int NVPWindow::sysExtensionSupported(const char *name){

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

 
  if (!exts) {
    return 0;
  }
  
  return stringInExtensionString(name,exts);

}

NVPWindow::NVPproc NVPWindow::sysGetProcAddress(const char *name){
    return (NVPWindow::NVPproc)glXGetProcAddress((const GLubyte *)name);
}

void NVPWindow::sysWaitEvents(){
    //Handle wait for XEvents.

}

static int getMilliCount(){
    timeb tb;
    ftime(&tb);

    int nCount = tb.millitm + (tb.time & 0xffffff) * 1000;
    return nCount;

}

static double s_frequency;
double NVPWindow::sysGetTime(){

    //Handle timeb get time.
    return 1.0/(double)getMilliCount();
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
	checkGL ( "glLinkProgram (init_screenquad)" );
	
	// Get texture parameter
	m_screenquad_utex1 = glGetUniformLocation (m_screenquad_prog, "uTex1" );
	m_screenquad_utex2 = glGetUniformLocation (m_screenquad_prog, "uTex2");
	m_screenquad_utexflags = glGetUniformLocation(m_screenquad_prog, "uTexFlags");
	m_screenquad_ucoords = glGetUniformLocation ( m_screenquad_prog, "uCoords" );
	m_screenquad_uscreen = glGetUniformLocation ( m_screenquad_prog, "uScreen" );


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
	if ( *glid == -1 ) glDeleteTextures ( 1, (GLuint*) glid );
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
	renderScreenQuadGL ( glid, -1, (float)0, (float)0, (float)getWidth(), (float)getHeight(), inv1); 
}

void NVPWindow::compositeScreenQuadGL(int glid1, int glid2, char inv1, char inv2)
{
	renderScreenQuadGL( glid1, glid2, (float)0, (float)0, (float)getWidth(), (float)getHeight(), inv1, inv2 );
}

void NVPWindow::renderScreenQuadGL ( int glid1, int glid2, float x1, float y1, float x2, float y2, char inv1, char inv2 )

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
	glEnable ( GL_TEXTURE_2D );
	glProgramUniform4f ( m_screenquad_prog, m_screenquad_ucoords, x1, y1, x2, y2 );
	glProgramUniform2f ( m_screenquad_prog, m_screenquad_uscreen, (float) getWidth(), (float) getHeight() );

	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture ( GL_TEXTURE_2D, glid1 );

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

	glProgramUniform1i(m_screenquad_prog, m_screenquad_utexflags, flags );	

	// Draw
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
	checkGL("glDraw");
	glUseProgram(0);

	glDepthMask(GL_TRUE);
}


void NVPWindow::sysSleep(double seconds){
    //handle process sleep.
    /*
        Uses Unistd takes ms as argument.
    */
    sleep(seconds);

}

void NVPWindow::sysInit(){
    //NOOP
}

void NVPWindow::sysDeinit(){
    //NOOP
}

static std::string s_path;
std::string NVPWindow::sysExePath(){
    return s_path;
}

void NVPWindow::sysVisibleConsole(){
    //Handle console stdout display.

}


// from file_png.cpp
extern void save_png ( char* fname, unsigned char* img, int w, int h, int ch );

void NVPWindow::save_frame ( char* fname )
{
	int w = getWidth();
	int h = getHeight();

	// Read back pixels
	unsigned char* pixbuf = (unsigned char*) malloc ( w*h*3 );

	glReadPixels ( 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixbuf );

	// Flip Y
	int pitch = w*3;
	unsigned char* buf = (unsigned char*) malloc ( pitch );
	for (int y=0; y < h/2; y++ ) {
		memcpy ( buf, pixbuf + (y*pitch), pitch );		
		memcpy ( pixbuf + (y*pitch), pixbuf + ((h-y-1)*pitch), pitch );		
		memcpy ( pixbuf + ((h-y-1)*pitch), buf, pitch );
	}

	// Save png
	save_png ( fname, pixbuf, w, h, 3 );

	free ( pixbuf );
	free ( buf );
}




bool WINinternal::create(const char *title,int width, int height){

  //XInitThreads();

  

    m_dpy = XOpenDisplay(0);
    char *dpyName = XDisplayName(NULL);

    //nvprintf("Display Name : %s\n",dpyName);

   Atom wmDelete;
 
    static int visual_attribs[] = {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE,8,
        GLX_GREEN_SIZE,8,
        GLX_BLUE_SIZE,8,
        GLX_ALPHA_SIZE,8,
        GLX_DEPTH_SIZE,24,
        //GLX_STENCIL_SIZE,8,
	//        GLX_SAMPLE_BUFFERS,1,
        //GLX_SAMPLES,8,
        GLX_DOUBLEBUFFER, True,
        None
    };

    int e1, e2;
    bool ret = glXQueryExtension(m_dpy, &e1, &e2);
    if ( !ret ) {
      nvprintf ( "No GLX available.\n" );
    } 

    int glxMajor,glxMinor;

    if(!glXQueryVersion(m_dpy,&glxMajor,&glxMinor) || ((glxMajor == 1) && (glxMinor < 3)) || (glxMajor < 1)){
      //nvprintf("Invalid GLX Version.\n");
        return false;
    }else{
      //nvprintf("GLX VERSION : %i.%i.\n",glxMajor, glxMinor);
    }

    int fbCount = 0;
    
    nvprintf("DPY : %p .\n",m_dpy); 
    GLXFBConfig *fbc = glXChooseFBConfig ( m_dpy, DefaultScreen(m_dpy), visual_attribs, &fbCount);

    if(!fbc){
      // nvprintf("Could not get FB config.\n");
      return false;
    }

    int bestfbc = -1,worstfbc = -1, best_num_sample = -1, worst_num_sample = 999;

    int i;

    for(i=0;i<fbCount;++i){
        XVisualInfo *vi = glXGetVisualFromFBConfig(m_dpy,fbc[i]);

        if(vi){
                int sampleBuf, samples;
                glXGetFBConfigAttrib(m_dpy,fbc[i],GLX_SAMPLE_BUFFERS, &sampleBuf);
                glXGetFBConfigAttrib(m_dpy,fbc[i],GLX_SAMPLES,&samples);

                printf("GLX Config Sample Buffers : %d.\n",sampleBuf);
                printf("GLX Config Samples : %d.\n",samples);

                if(bestfbc < 0 || sampleBuf && samples > best_num_sample) bestfbc = i,best_num_sample = samples;
                if(worstfbc < 0 || !sampleBuf || samples < worst_num_sample) worstfbc = i, worst_num_sample = samples;
         }
        XFree(vi);
    }

    m_glx_fb_config = fbc[bestfbc];

    XFree(fbc);

    XVisualInfo *vi = glXGetVisualFromFBConfig(m_dpy,m_glx_fb_config);

    XSetWindowAttributes swa;
    Colormap cmap;

    swa.colormap = cmap = XCreateColormap(m_dpy,RootWindow(m_dpy,vi->screen),vi->visual,AllocNone);
    swa.background_pixmap = None;
    swa.border_pixel = 0;
    swa.event_mask = 
    ExposureMask | 
    KeyPressMask | 
    KeyReleaseMask | 
    ButtonPressMask | 
    ButtonReleaseMask |
    PointerMotionMask |
    StructureNotifyMask;
    swa.override_redirect = False;


    printf("Creating Window.\n");

    printf("Width : %d Height : %d.\n",width,height);

    m_window = XCreateWindow(m_dpy,RootWindow(m_dpy,vi->screen),0,0,width,height,0,vi->depth,InputOutput,vi->visual,
        CWBorderPixel | CWColormap | CWEventMask | CWOverrideRedirect,&swa);
    sleep(1);

    printf("Window : %d.\n",m_window);
    XFree(vi);
    
    XSetStandardProperties(m_dpy,m_window,title,title,None,NULL,0,NULL);
    wmDelete = XInternAtom(m_dpy,"WM_DELETE_WINDOW",True);
    XSetWMProtocols(m_dpy,m_window,&wmDelete,1);
    XMapRaised(m_dpy,m_window);
    XFlush(m_dpy);


    printf("Window Created.\n");
    return true;

}


int main(int argc, char **argv){

nvprintf ("Starting here\n");
Display *dpy = XOpenDisplay(0); 
int nelements;
GLXFBConfig *fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), 0, &nelements);
nvprintf ("Got here\n");

    std::string exe = std::string(argv[0]);
    std::replace(exe.begin(),exe.end(),'\\','/');

    size_t last = exe.rfind('/');
    if(last != std::string::npos){
        //s_path = exe.substr(0,last) + std::string("/");


    }
    printf("Sys Exe Path : %s.\n",s_path.c_str());

    sample_main(argc,(const char **)&argv[0]);

    for(int i=0;i<g_windows.size();++i){
            NVPWindow *pWin = g_windows[i];
            if(pWin->m_internal){
                if(pWin->m_internal->m_glx_context != NULL){
                    //Handle GLX cleanup

                }
                delete pWin->m_internal;
            }

    }

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
    //char *prefix = "";
    //switch(level)
    //{
    //case LOGLEVEL_WARNING:
    //    prefix = "LOG *WARNING* >> ";
    //    break;
    //case LOGLEVEL_ERROR:
    //    prefix = "LOG **ERROR** >> ";
    //    break;
    //case LOGLEVEL_OK:
    //    prefix = "LOG !OK! >> ";
    //    break;
    //case LOGLEVEL_INFO:
    //default:
    //    break;
    //}
#ifdef WIN32
    //OutputDebugStringA(prefix);
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
bool getFileLocation ( char* filename, char* outpath )
{
	std::vector<std::string> paths;
	paths.push_back ( "./");
	paths.push_back ( ASSET_PATH );
	bool result = getFileLocation ( filename, outpath, paths );
	return result;
}

bool getFileLocation ( char* filename, char* outpath, std::vector<std::string> searchPaths )
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
//------------------------------------------------------------------------------
