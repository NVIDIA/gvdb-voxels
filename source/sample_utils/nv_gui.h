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

//
// Functionality in this file:
//  - nvMesh: Construct, load, and render meshes. PLY format supported
//  - nvImg: Cosntruct, load, and render images. PNG and TGA format supported
//  - nvDraw: A lightweight, efficient, 2D drawing API. Uses VBOs to render
//    lines, circles, triangles, and text. Allows for both static and dynamic 
//    groups (rewrite per frame), and immediate style (out-of-order) calling.
//  - nvGui: A lightweight class for creating on-screen GUIs. Currently only checkboxes
//    or sliders supported. Relies on nvDraw to render.
//

#ifndef DEF_NV_GUI
	#define DEF_NV_GUI

	#include <GL/glew.h>
	#include <string>
	#include <vector>
    typedef	GLuint			BUF;		
	typedef	GLuint			TEX;
	typedef unsigned int	uint;
	#define UINT_NULL	0xFFFFFFFF
	#define IDX_NULL	0xFF000000		// OpenGL - primitive restart index

	#ifdef USE_GVDB
		#include "gvdb_camera.h"
		using namespace nvdb;
	#else
		#include "camera.h"
	#endif


	//-------------------------------------- PLY FORMAT
	struct PlyProperty {            // PLY Format structures
        char                        type;
        std::string                 name;
    };
    struct PlyElement {
        int							num;
        char                        type;        // 0 = vert, 1 = face
        std::vector<PlyProperty>    prop_list;
    };
	typedef unsigned long			xref;

	#define PLY_UINT        0                    // PLY Format constants
    #define PLY_INT         1
    #define PLY_FLOAT       2
    #define PLY_LIST        3
    #define PLY_VERTS       4
    #define PLY_FACES       5

	#define attrPos			3
	#define attrClr			4
	#define attrUV			5
	#define attrPos2		6

	struct Vertex {
		float	x, y, z;
		float	nx, ny, nz;
		float	tx, ty, tz;
	};

	//------------------------------------ nvMesh
	class nvMesh {
	public:
		nvMesh();

		void Clear ();
		void AddPlyElement ( char typ, int n );
		void AddPlyProperty ( char typ, std::string name );
		int AddVert ( float x, float y, float z, float tx, float ty, float tz );
		void SetVert ( int n, float x, float y, float z, float tx, float ty, float tz );
		void SetNormal ( int n, float x, float y, float z );
		int AddFace ( int v0, int v1, int v2 );
		int AddFace4 ( int v0, int v1, int v2, int v3 );
		int FindPlyElem ( char typ );
		int FindPlyProp ( int elem, std::string name );
		bool LoadPly ( char* fname, float scal );	// Load PLY
		void ComputeNormals ();						// Compute surface normals
		void UpdateVBO ( bool rebuild, int cnt = 1);				// Update GPU buffers
		void SelectVBO ();
		void SelectVAO ();
		void Draw ( int inst );						// Draw
		void DrawPatches ( int inst );

		int getNumFaces ()		{ return (int) mNumFaces; }		
		int getNumVerts ()		{ return (int) mVertices.size(); }

	private:

		std::vector<Vertex>		mVertices;			// Vertices
		std::vector<unsigned int> mFaceVN;			// Face data (must be uniformly 3 or 4 sided)
		int						mNumFaces;			// Num faces		
		int						mNumSides;			// Must be 3 or 4		
		std::vector<BUF>		mVBO;				// Buffers
		BUF						mVAO;

        std::vector< PlyElement* >    m_Ply;		// Ply loading
        int						m_PlyCurrElem;
		int						localPos, localNorm, localUV, localPos2;
	};

	//------------------------------------ nvImg
	#define IMG_RGB			0
	#define IMG_RGBA		1
	#define IMG_GREY16		2	
	
	class nvImg {
	public:
		nvImg();
		void	Create ( int x, int y, int fmt );
		void	Fill ( float r, float g, float b, float a );
		bool	LoadPng ( char* fname, bool bGray=false );		// Load PNG
		bool	LoadTga ( char* fname );			// Load TGA
		void	FlipY ();
		void	SavePng ( char* fname );
		void	UpdateTex ();						// Update GPU texture
		void	BindTex ();							// Bind GPU texture
		TEX		getTex()	{ return mTex;}
		int		getSize ()	{ return mSize;}
		int		getWidth()	{ return mXres;}
		int		getHeight()	{ return mYres;}		
		unsigned char* getData()	{ return mData;}

	private:
		int						mXres, mYres;
		int						mSize, mFmt;
		unsigned char*			mData;

		TEX						mTex;
		#ifdef USE_DX
			ID3D11ShaderResourceView*	mTexIV;
		#endif
	};

	//-------------------------------------- Fonts (Tristan Lorach)
	// Read from .bin files	
    struct GlyphInfo {
       struct Pix {			// pixel oriented data       
           int u, v;
           int width, height;
           int advance;
           int offX, offY;
       };
       struct Norm {		// normalized data       
           float u, v;		// position in the map in normalized coords
           float width, height;
           float advance;
           float offX, offY;
       };
       Pix  pix;
       Norm norm;
    };
    struct FileHeader {
       int texwidth, texheight;
       struct Pix {
           int ascent;
           int descent;
           int linegap;
       };
       struct Norm {
           float ascent;
           float descent;
           float linegap;
       };
       Pix  pix;
       Norm norm;
       GlyphInfo glyphs[256];
    };

	#define GRP_TRI			0
	#define GRP_TRITEX		1
	#define GRP_LINES		2	
	#define GRP_IMG			3
	#define GRP_BOX			4
	#define GRP_MAX			5

	#ifndef xlong
		typedef unsigned long long	xlong;		// 64-bit integer
	#endif

	//-------------------------------------- 2D DRAW SETS
	struct nvVert {
		float	x, y, z;
		float	r, g, b, a;
		float	tx, ty;
	};
	struct nvSet {
		nvSet () { mImgs=0; for (int n=0; n<GRP_MAX; n++) {mVBO[n]=0; mVBOI[n]=0; mNum[n]=0; mMax[n]=0; mGeom[n]=0; } }
		float	model[16];
		float	view[16];
		float	proj[16];		
		float	zfactor;
		xlong	mNum[GRP_MAX];
		xlong	mMax[GRP_MAX];			
		xlong	mNumI[GRP_MAX];
		xlong	mMaxI[GRP_MAX];		
		nvVert*	mGeom[GRP_MAX];			// Geom data
		uint*	mIdx[GRP_MAX];			// Index data
		nvImg**	mImgs;					// Image data

		BUF		mVBO[GRP_MAX];			// GPU handles
		BUF		mVBOI[GRP_MAX];
	};

	//-------------------------------------- nvDraw (2D Drawing)
	#define SCOLOR	0		// Color shader (2D & 3D)
	#define SINST	1		// Instance shader
	#define S3D		2		// 3D shader (Phong)

	#define localPos	0
	#define	localClr	1 
	#define localUV		2
	#define localNorm	3

	class nvDraw {
	public:		
		nvDraw ();
		void drawGL ();
		void setView2D ( int w, int h );
		void setView2D ( float* model, float* view, float* proj );
		void setOrder2D ( bool zt, float zfactor );
		void updateStatic2D ( int n );
		void start2D () { start2D (false); }
		int  start2D ( bool bStatic );
		//void assignMVP ( float* model, float* view, float* proj );		
		nvVert* allocGeom ( int cnt, int grp, nvSet* set, int& ndx );	// alloc geom for lines/tri/text
		nvVert* allocGeom ( nvImg* img, int grp, nvSet* set, int& ndx );			// alloc geom for drawing images
		uint*    allocIdx  ( int cnt, int grp, nvSet* set );
		void end2D ();
		void remove2D ( int id );
		void setText ( float scale, float kern )		{ mTextScale = scale; mTextKern = kern; }		
		void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawImg ( nvImg* img, float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a );
		void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a );
		float getTextX ( char* msg );
		float getTextY ( char* msg );		
		void draw2D ();		// do all 2D draws
		
		void setView3D ( nvSet& s, Camera3D* cam );
		void start3D ( Camera3D* cam );			
		void drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a );
		void drawBox3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a );
		void end3D ();
		void draw3D ();		// do all 3D draws
		
		bool Initialize ( const char* fontName );
		bool LoadFont (const char * fontName );
		void CreateSColor ();		
		void CreateSInst ();
		void CreateS3D ();
		void UpdateVBOs ( nvSet& s );
		void SetDefaultView ( nvSet& s, float w, float h, float zf );
		void SetMatrixView ( nvSet& s, float* model, float* view, float* proj, float zf );
		void drawSet2D ( nvSet& buf );
		void drawSet3D ( nvSet& buf );
		void setLight (int s, float x1, float y1, float z1 );
		
	private:
		std::vector<nvSet>	mStatic;		// static 2D draw - saved frame-to-frame
		std::vector<nvSet>	mDynamic;		// dynamic 2D draw - discarded each frame
		std::vector<nvSet>  m3D;			// 3D draw		
		int					mCurrZ;
		int					mCurr, mDynNum, m3DNum;
		nvSet*				mCurrSet;
		nvSet				mBoxSet;		// box geometry
		float				mWidth, mHeight;
		float				mTextScale, mTextKern;
		float				mModelMtx[16], mViewMtx[16], mProjMtx[16];		
		double				mZFactor;
		
	#ifdef USE_DX
		ID3D11Buffer*		mpMatrixBuffer[3];		// 2D model/view/proj
		ID3D11VertexShader* mVS;		// 2D shader
		ID3D11PixelShader*  mPS;
		ID3D11InputLayout*  mLO;		// 2D layout

	#else
		GLuint				mSH[3];					// Shaders		
		GLint				mModel[3], mProj[3];	// Shader parameters
		GLint				mView[3], mFont[3];
		GLint				mLight[3];
		GLuint				mVAO;
	#endif

		nvImg				mFontImg, mWhiteImg;
		FileHeader			mGlyphInfos;
	};
	
	//-------------------------------------- nvGui (2D GUIS)

	#define GUI_PRINT		0
	#define GUI_SLIDER		1
	#define GUI_CHECK		2
	#define GUI_TEXT		3
	#define GUI_COMBO		4
	#define GUI_TOOLBAR		5
	#define GUI_ICON		6

	#define GUI_NULL		255
	#define GUI_BOOL		0
	#define GUI_INT			1
	#define GUI_FLOAT		2
	#define GUI_STR			3	
	#define	GUI_VEC3		4

	struct Gui {
		float			x, y, w, h;
		int				gtype;
		std::string		name;
		int				dtype;
		void*			data;
		float			vmin, vmax;
		int				size;
		bool			changed;
		Vector4DF		backclr;
		std::vector<std::string>	items;
		std::vector<nvImg*>			imgs;
	};

	class nvGui {
	public:
		nvGui ();
		int		AddGui ( float x, float y, float w, float h, const char* name, int gtype, int dtype, void* data, float vmin, float vmax );
		int		AddItem ( char* name, char* imgname = 0x0 );
		void    SetBackclr ( float r, float g, float b, float a );
		bool	guiChanged ( int n );
		std::string	getItemName ( int g, int v );		
		bool	MouseDown ( float x, float y );
		bool	MouseUp ( float x, float y );
		bool	MouseDrag ( float x, float y );
		void	Draw ( nvImg* img );
		void	Clear ();

	private:
		std::vector<Gui>	mGui;
		int					mActiveGui;
	};

	// Public functions
	//
	extern nvDraw	g_2D;
	extern bool init2D ( const char* fontName );
	extern void start2D ();
	extern void start2D (bool bStatic);
	extern void static2D ();
	extern void end2D ();
	extern void draw2D ();
	extern void setview2D ( int w, int h );
	extern void setview2D ( float* model, float* view, float* proj );
	extern void setorder2D ( bool zt, float zfactor );
	extern void updatestatic2D ( int n );
	extern void setText ( float scale, float kern );
	extern float getTextX ( char* msg );
	extern float getTextY ( char* msg );
	extern void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawImg ( nvImg* img, float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a );
	extern void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a );

	extern void start3D ( Camera3D* cam );
	extern void drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a );
	extern void drawBox3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a );
	extern void end3D ();
	extern void draw3D ();
	extern void drawGL ();
	extern void setLight (int s, float x1, float y1, float z1 );

	typedef void (*CallbackFunc)(int, float);
	extern nvGui	g_Gui;
	extern void		drawGui ( nvImg* img );	
	extern void     clearGuis ();
	extern int		addGui ( int x, int y, int w, int h, const char* name, int gtype, int dtype, void* data, float vmin, float vmax );
	extern int		addItem ( char* name, char* imgname = 0x0 );
	extern void		setBackclr ( float r, float g, float b, float a );
	extern std::string guiItemName ( int n, int v );
	extern bool		guiHandler ( int button, int action, int x, int y );	
	extern bool		guiMouseDown ( float x, float y );
	extern bool		guiMouseUp ( float x, float y );
	extern bool		guiMouseDrag ( float x, float y );
	extern bool		guiChanged ( int n );
	extern void		guiSetCallback ( CallbackFunc f );
	
#endif