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

/*!
 * This file provides a utility classes for 2D Drawing and GUIs
 * Functionality in this file:
 *  - nvMesh: Construct, load, and render meshes. PLY format supported
 *  - nvImg: Cosntruct, load, and render images. PNG and TGA format supported
 *  - nvDraw: A lightweight, efficient, 2D drawing API. Uses VBOs to render
 *     lines, circles, triangles, and text. Allows for both static and dynamic 
 *     groups (rewrite per frame), and immediate style (out-of-order) calling.
 *  - nvGui: A lightweight class for creating on-screen GUIs. Currently only checkboxes
 *    or sliders supported. Relies on nvDraw to render.
 * Useage: 
 *    1. Main programs implement the functions required by app_opengl/directx.
 *    2. During display(), first do any rendering desired by your demo or application.
 *    3. Then call drawGui to render GUI items to the 2D layer.
 *    4. Then call draw2D to actually render all 2D objects (gui and user-specified)
 *    5. Finally call SwapBuffers 
 */

#include "nv_gui.h"
#include "main.h"			// for checkGL and nvprintf
#include "file_png.h"		// for png fonts
#include "file_tga.h"		// for tga fonts

#ifdef USE_GVDB
	#include "gvdb_vec.h"	
	using namespace nvdb;
#else
	#include "vec.h"
#endif

// Globals
typedef void (*CallbackFunc)(int, float);
nvDraw			g_2D;
nvGui			g_Gui;
CallbackFunc	g_GuiCallback;

struct MatrixBuffer 
{
    float m[16];    
}; 

// Utility functions
bool init2D ( const char* fontName )		{ return g_2D.Initialize( fontName ); }
void drawGL ()		{ g_2D.drawGL(); }
void setLight (int s, float x1, float y1, float z1 )	{ g_2D.setLight(s,x1,y1,z1); }
void start2D ()		{ g_2D.start2D(); }
void start2D (bool bStatic)		{ g_2D.start2D(bStatic); }
void updatestatic2D ( int n )	{ g_2D.updateStatic2D (n); }
void static2D ()	{ g_2D.start2D(true); }
void end2D ()		{ g_2D.end2D(); }
void draw2D ()		{ g_2D.draw2D(); }
void setview2D ( int w, int h)			{ g_2D.setView2D( w,h ); }
void setview2D ( float* model, float* view, float* proj )		{ g_2D.setView2D( model, view, proj ); }
void setorder2D ( bool zt, float zfactor )		{ g_2D.setOrder2D( zt, zfactor ); }
void setText ( float scale, float kern )	{ g_2D.setText(scale,kern); }
void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawLine(x1,y1,x2,y2,r,g,b,a); }
void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawRect(x1,y1,x2,y2,r,g,b,a); }
void drawImg ( nvImg* img,  float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawImg ( img, x1,y1,x2,y2,r,g,b,a ); }
void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawFill(x1,y1,x2,y2,r,g,b,a); }
void drawTri  ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )	{ g_2D.drawTri(x1,y1,x2,y2,x3,y3,r,g,b,a); }
void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )		{ g_2D.drawCircle(x1,y1,radius,r,g,b,a); }
void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D.drawCircleDash(x1,y1,radius,r,g,b,a); }
void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D.drawCircleFill(x1,y1,radius,r,g,b,a); }
void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )				{ g_2D.drawText(x1,y1,msg,r,g,b,a); }
float getTextX ( char* msg )	{ return g_2D.getTextX(msg); }
float getTextY ( char* msg )	{ return g_2D.getTextY(msg); }

void start3D ( Camera3D* cam )	{ g_2D.start3D( cam ); }
void drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a ) { g_2D.drawLine3D(x1,y1,z1,x2,y2,z2,r,g,b,a); }
void drawBox3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a ) { g_2D.drawBox3D(x1,y1,z1,x2,y2,z2,r,g,b,a); }
void end3D ()		{ g_2D.end3D(); }
void draw3D ()		{ g_2D.draw3D(); }

void drawGui ( nvImg* img)		{ g_Gui.Draw( img ); }
void clearGuis ()				{ g_Gui.Clear(); }
int  addGui ( int x, int y, int w, int h, const char* name, int gtype, int dtype, void* data, float vmin, float vmax ) { return g_Gui.AddGui ( float(x), float(y), float(w), float(h), name, gtype, dtype, data, vmin, vmax ); }
void setBackclr ( float r, float g, float b, float a )	{ return g_Gui.SetBackclr ( r,g,b,a ); }
int  addItem ( char* name )		{ return g_Gui.AddItem ( name ); }
int  addItem ( char* name, char* imgname ) { return g_Gui.AddItem ( name, imgname ); }
std::string guiItemName ( int n, int v )	{ return g_Gui.getItemName ( n, v ); }
bool guiChanged ( int n )  { return g_Gui.guiChanged(n); }
bool guiMouseDown ( float x, float y )	{ return g_Gui.MouseDown(x,y); }
bool guiMouseUp ( float x, float y )	{ return g_Gui.MouseUp(x,y); }
bool guiMouseDrag ( float x, float y )	{ return g_Gui.MouseDrag(x,y); }
void guiSetCallback ( CallbackFunc f )  { g_GuiCallback = f; }



bool guiHandler ( int button, int action, int x, int y )
{
	switch ( action ) {
	case 0:		return g_Gui.MouseUp( float(x), float(y) );		break;
	case 1:		return g_Gui.MouseDown( float(x), float(y) );	break;
	case 2:		return g_Gui.MouseDrag( float(x), float(y) );	break;
	}
	return false;
}

nvDraw::nvDraw ()
{
	mCurrZ = 0;	
	mDynNum = 0;
	mTextScale = 1;
	mTextKern = 0;
	mWidth = 0;
	mHeight = 0;
	mVAO = 65535;
	setOrder2D ( false, 1.0 );
}

void nvDraw::setView2D ( int w, int h )
{
	mWidth = float(w); mHeight = float(h);
}
void nvDraw::setView2D ( float* model, float* view, float* proj )
{
	mWidth = -1; mHeight = -1;

	memcpy ( mModelMtx, model, 16 * sizeof(float) );
	memcpy ( mViewMtx, view, 16 * sizeof(float) );
	memcpy ( mProjMtx, proj, 16 * sizeof(float) );
}
void nvDraw::updateStatic2D ( int n )
{
	if ( n < 0 || n >= mStatic.size()) return;
	if ( mWidth==-1 ) {		
		SetMatrixView ( mStatic[n], mModelMtx, mViewMtx, mProjMtx, (float) mZFactor );
	} else {
		SetDefaultView ( mStatic[n], mWidth, mHeight, (float) mZFactor );
	}
}
void nvDraw::setView3D ( nvSet& s, Camera3D* cam )
{
	Matrix4F ident; 
	ident.Identity ();
	memcpy ( s.model, ident.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.view, cam->getViewMatrix().GetDataF(), 16 * sizeof(float) );
	memcpy ( s.proj, cam->getProjMatrix().GetDataF(), 16 * sizeof(float) );
}

void nvDraw::start3D ( Camera3D* cam )
{
	if ( mVAO == 65535 ) {
		nvprintf ( "ERROR: nv_gui was not initialized. Must call init2D.\n" );
		nverror ();
	}
	if ( m3DNum >= m3D.size() ) {
		nvSet new_set;	
		memset ( &new_set, 0, sizeof(nvSet) );
		m3D.push_back ( new_set );		
	}	
	nvSet* s = &m3D[ m3DNum ];	
	m3DNum++;
	mCurrSet = s;	
	s->zfactor = -1;
	for (int n=0; n < GRP_MAX; n++ ) {
		s->mNum[n] = 0;	
		s->mNumI[n] = 0;	
	}
	setView3D ( *s, cam );		// set 3D transforms
}		

void nvDraw::drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 2, GRP_LINES, mCurrSet, ndx );
	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = z2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;
}
void nvDraw::drawBox3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 2, GRP_BOX, mCurrSet, ndx );	
	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = z2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;
}

void nvDraw::end3D ()
{
	mCurrSet = 0x0;	
}

int nvDraw::start2D ( bool bStatic )
{
	if ( mVAO == 65535 ) {
		nvprintf ( "ERROR: nv_gui was not initialized. Must call init2D.\n" );
		nverror ();
	}
	nvSet new_set;
	nvSet* s;
	
	if ( bStatic ) {		
		mStatic.push_back ( new_set );		
		s = &mStatic[ mStatic.size()-1 ]; 		
		for (int n=0; n < GRP_MAX; n++ ) s->mGeom[n] = 0x0;
		s->mImgs = 0x0;
		mCurrSet = s;		
	} else {
		int curr = mDynNum;					
		if ( mDynNum >= mDynamic.size() ) {			
			mDynamic.push_back ( new_set );						
			mDynNum = (int) mDynamic.size();
			s = &mDynamic[curr];	
			for (int n=0; n < GRP_MAX; n++ ) s->mGeom[n] = 0x0;
			s->mImgs = 0x0;
		} else {		
			mDynNum++;
			s = &mDynamic[curr];	
		}		
		mCurrSet = s;
	}
	for (int n=0; n < GRP_MAX; n++ ) {
		s->mNum[n] = 0;		s->mMax[n] = 0;
		s->mNumI[n] = 0;	s->mMaxI[n] = 0;	s->mIdx[n] = 0;		
	}
	if ( mWidth==-1 ) {
		SetMatrixView ( *s, mModelMtx, mViewMtx, mProjMtx, (float) mZFactor );
	} else {
		SetDefaultView ( *s, mWidth, mHeight, (float) mZFactor );
	}

	return mCurr;
}

void nvDraw::setOrder2D ( bool zt, float zfactor )
{	
	if ( zt == false ) zfactor = 1;
	mZFactor = zfactor;
}

void nvDraw::SetDefaultView ( nvSet& s, float w, float h, float zf )
{
	s.zfactor = zf;

	Matrix4F proj, view, model;
	view.Scale ( 2.0/w, -2.0/h, s.zfactor );	
	model.Translate ( -w/2.0, -h/2.0, 0 );
	view *= model;	
	model.Identity ();
	proj.Identity ();

	memcpy ( s.model, model.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.view,  view.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.proj,  proj.GetDataF(), 16 * sizeof(float) );	
}
void nvDraw::SetMatrixView ( nvSet& s, float* model, float* view, float* proj, float zf )
{
	s.zfactor = zf;
	memcpy ( s.model, model, 16 * sizeof(float) );
	memcpy ( s.view,  view, 16 * sizeof(float) );
	memcpy ( s.proj,  proj, 16 * sizeof(float) );	
}

void nvDraw::end2D ()
{
	mCurrSet = 0x0;	
}

nvVert* nvDraw::allocGeom ( int cnt, int grp, nvSet* s, int& ndx )
{
	if ( s == 0x0 ) {
		nvprintf ( "ERROR: draw must be inside of draw2D/end2D or draw3D/end3D\n" );
		nverror ();
	}
	if ( s->mNum[grp] + cnt >= s->mMax[grp] ) {		
		xlong new_max = s->mMax[grp] * 8 + cnt;		
		//	nvprintf  ( "allocGeom: expand, %lu\n", new_max );
		nvVert* new_data = (nvVert*) malloc ( new_max*sizeof(nvVert) );
		if ( s->mGeom[grp] != 0x0 ) {
			memcpy ( new_data, s->mGeom[grp], s->mNum[grp]*sizeof(nvVert) );
			free ( s->mGeom[grp] );
		}
		s->mGeom[grp] = new_data;
		s->mMax[grp] = new_max;
	}
	nvVert* start = s->mGeom[grp] + s->mNum[grp];
	ndx = (int) s->mNum[grp];
	s->mNum[grp] += cnt;	
	return start;
}

nvVert* nvDraw::allocGeom ( nvImg* img, int grp, nvSet* s, int& ndx )
{
	int cnt = 4;	
	
	if ( s->mNum[grp] + cnt >= s->mMax[grp] ) {
		xlong new_max = s->mMax[grp] * 8 + cnt;		
		nvImg** new_imgs = (nvImg**) malloc ( new_max*sizeof(nvImg*) );
		nvVert* new_data = (nvVert*) malloc ( new_max*sizeof(nvVert) );
		if ( s->mGeom[grp] != 0x0 ) {
			memcpy ( new_data, s->mGeom[grp], s->mNum[grp]*sizeof(nvVert) );
			memcpy ( new_imgs, s->mImgs, s->mNum[grp]*sizeof(nvImg*) );
			free ( s->mGeom[grp] );
			free ( s->mImgs );
		}
		s->mGeom[grp] = new_data;		
		s->mMax[grp] = new_max;
		s->mImgs = new_imgs;
	}
	nvVert* start = s->mGeom[grp] + s->mNum[grp];
	s->mImgs[ s->mNum[grp] / cnt ] = img;		// stride is 4 corners
	ndx = (int) s->mNum[grp];
	s->mNum[grp] += cnt;		// 4 corners per img
	return start;
}

uint* nvDraw::allocIdx ( int cnt, int grp, nvSet* s )
{
	if ( s->mNumI[grp] + cnt >= s->mMaxI[grp] ) {		
		xlong new_max = s->mMaxI[grp] * 8 + cnt;
		// nvprintf  ( "allocIdx: expand, %lu\n", new_max );
		uint* new_data = (uint*) malloc ( new_max*sizeof(uint) );
		if ( s->mIdx[grp] != 0x0 ) {
			memcpy ( new_data, s->mIdx[grp], s->mNumI[grp]*sizeof(uint) );
			delete s->mIdx[grp];
		}
		s->mIdx[grp] = new_data;
		s->mMaxI[grp] = new_max;
	}
	uint* start = s->mIdx[grp] + s->mNumI[grp];		
	s->mNumI[grp] += cnt;
	return start;
}

void nvDraw::remove2D ( int id )
{


}
void nvDraw::drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw line.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 2, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}
void nvDraw::drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw rect.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 8, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}

void nvDraw::drawImg ( nvImg* img, float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( img, GRP_IMG, mCurrSet, ndx );
	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = -a; v->tx = 0; v->ty = 1;	v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = -a; v->tx = 1; v->ty = 1;	v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = -a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = -a; v->tx = 1; v->ty = 0;
}

void nvDraw::drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw fill.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 4, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 5, GRP_TRI, mCurrSet );

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 0;	v++;
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 1;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1;
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
}
void nvDraw::drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw tri.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 3, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 4, GRP_TRI, mCurrSet );

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x3; v->y = y3; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;		
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
}

void nvDraw::drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw circle.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 62, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy, dxl, dyl;	
	dxl = (float) cos( (0/31.0)*3.141592*2.0 )*radius;
	dyl = (float) sin( (0/31.0)*3.141592*2.0 )*radius;
	for (int n=1; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1+dxl; v->y = y1+dyl; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		dxl = dx; dyl = dy;
	}		
}
void nvDraw::drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 32, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy;		
	for (int n=0; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;		
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;		
	}		
}
void nvDraw::drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 64, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 65, GRP_TRI, mCurrSet );
	
	float dx, dy;
	for (int n=0; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
	}	
	*i++ = IDX_NULL;
}

// from Tristan Lorach, OpenGLText
void nvDraw::drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	nvprintf  ( "Draw text.\n" );
#endif
	int len = (int) strlen ( msg );
	if ( len == 0 ) 
		return;

#ifdef DEBUG_UTIL
	if ( len > 128 ) {
		MessageBox ( NULL, "Error: drawText over 128 chars\n", "drawText", MB_OK );
		exit(-1);
	}
#endif
	int ndx;
	nvVert* v = allocGeom ( len*4, GRP_TRITEX, mCurrSet, ndx );
	uint* i = allocIdx ( len*5, GRP_TRITEX, mCurrSet );

	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = x1+1;
	float lPosY = y1;

	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;
	int cnt = 0;
	
	while (*c != '\0' && cnt < len ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;
			
			v->x = pX; v->y = pY; v->z = 0; 
			v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;

			v->x = pX; v->y = pY - gly.pix.height*mTextScale; v->z = 0; 
			v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u; v->ty = gly.norm.v + gly.norm.height;	v++;
			
			v->x = pX + gly.pix.width*mTextScale; v->y = pY; v->z = 0; 
			v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v;	v++;
			
			v->x = pX + gly.pix.width*mTextScale; v->y = pY - gly.pix.height*mTextScale; v->z = 0; 
			v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;

			*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
			cnt++;
		}
		c++;
	}

	for (int n=cnt; n < len; n++ ) {
		v->x = 0; v->y = 0; v->z = 0; v->r = 0; v->g = 0; v->b = 0; v->a = 0;	v->tx = 0; v->ty = 0; v++;
		v->x = 0; v->y = 0; v->z = 0; v->r = 0; v->g = 0; v->b = 0; v->a = 0;	v->tx = 0; v->ty = 0; v++;
		v->x = 0; v->y = 0; v->z = 0; v->r = 0; v->g = 0; v->b = 0; v->a = 0;	v->tx = 0; v->ty = 0; v++;
		v->x = 0; v->y = 0; v->z = 0; v->r = 0; v->g = 0; v->b = 0; v->a = 0;	v->tx = 0; v->ty = 0; v++;
		*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
	}
}

float nvDraw::getTextX ( char* msg )
{
	int len = (int) strlen ( msg );
	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = 0;
	float lPosY = 0;
	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;	
	while (*c != '\0' ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
		}
		c++;
	}
	return lPosX;
}
float nvDraw::getTextY ( char* msg )
{
	int len = (int) strlen ( msg );
	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = 0;
	float lPosY = 0;
	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;	
	while (*c != '\0' ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
		}
		c++;
	}
	return lPosY;
}

void nvDraw::CreateSColor ()
{
	#ifdef USE_DX

		// DirectX - Create shaders
		CHAR* g_strVS = 
			"cbuffer MatrixBuffer : register( b0 ) { row_major matrix Model; }\n"
			"cbuffer MatrixBuffer : register( b1 ) { row_major matrix View; }\n"
			"cbuffer MatrixBuffer : register( b2 ) { row_major matrix Proj; }\n"			
			"\n"
			"struct VS_IN { \n"
			"   float3 pos:POSITION; float4 clr:COLOR; float2 tex:TEXCOORD;\n"			
			//"   matrix instmodel: WORLDVIEW; \n"
			"}; \n"
			"struct VS_OUT { float4 pos:SV_POSITION; float4 color:COLOR; float2 tex:TEXCOORD0; }; \n"			
			"VS_OUT VS (VS_IN input, unsigned int InstID : SV_InstanceID ) { \n"
			"   VS_OUT output = (VS_OUT) 0;\n"			
			"   output.color = input.clr; \n"
			"   output.tex = input.tex;\n"
			"   output.pos = mul ( mul ( mul ( float4(input.pos,1), Model ), View ), Proj );\n"			
			"   return output;\n"
			"}\n";

		 CHAR *g_strPS = 			
			"Texture2D intex;  \n"
			"SamplerState sampleLinear { Filter = MIN_MAG_MIP_LINEAR; AddressU = Wrap; AddressV = Wrap; };\n"
			"struct PS_IN { float4 pos:SV_POSITION; float4 color:COLOR; float2 tex:TEXCOORD0; }; \n"
			"float4 PS ( PS_IN input ) : SV_Target\n"
			"{\n"
			//"    return  input.color;\n"
			"    return input.color * float4( 1, 1, 1, intex.Sample ( sampleLinear, input.tex ).x) ;\n"
			"}\n";

		DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
		#ifdef _DEBUG
			dwShaderFlags |= D3D10_SHADER_DEBUG;
		#endif
		ID3D10Blob* pBlobVS = NULL;
		ID3D10Blob* pBlobError = NULL;
		ID3D10Blob* pBlobPS = NULL;

		// Create vertex shader
		HRESULT hr = D3DCompile( g_strVS, lstrlenA( g_strVS ) + 1, "VS", NULL, NULL, "VS", "vs_4_0", dwShaderFlags, 0, &pBlobVS, &pBlobError );
		checkSHADER ( hr, pBlobError );	
		checkHR ( g_pDevice->CreateVertexShader( pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), NULL, &mVS ), "CreateVertexShader" );
    	
		// Create pixel shader
		hr = D3DCompile( g_strPS, lstrlenA( g_strPS ) + 1, "PS", NULL, NULL, "PS", "ps_4_0", dwShaderFlags, 0, &pBlobPS, &pBlobError ) ;		
		checkSHADER ( hr, pBlobError );
		checkHR ( g_pDevice->CreatePixelShader( pBlobPS->GetBufferPointer(), pBlobPS->GetBufferSize(), NULL, &mPS ), "CreatePixelShader" );
		
		// Create input-assembler layout
		D3D11_INPUT_ELEMENT_DESC vs_layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,		0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,	1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,			2, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		};
		UINT numElements = sizeof( vs_layout ) / sizeof( vs_layout[0] );		
		checkHR ( g_pDevice->CreateInputLayout( vs_layout, numElements, pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), &mLO ), "CreateInputLayout"  );

	#else
		// OpenGL - Create shaders
		char buf[16384];
		int len = 0;
		checkGL( "Start shaders" );

		// OpenGL 4.2 Core
		// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		GLchar const * vss =
			"#version 420\n"
			"\n"
			"layout(location = 0) in vec3 inPosition;\n"
			"layout(location = 1) in vec4 inColor;\n"
			"layout(location = 2) in vec2 inTexCoord;\n"
			"out vec3 position;\n"		
			"out vec4 color;\n"				
			"out vec2 texcoord;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"		
			"out gl_PerVertex {\n"
			"   vec4 gl_Position;\n"		
			"};\n"
			"\n"
			"void main()\n"
			"{\n"		
			"	 position = (modelMatrix * vec4(inPosition,1)).xyz;\n"
			"    color = inColor;\n"
			"    texcoord = inTexCoord;\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(inPosition,1);\n"
			"}\n"
		;
		glShaderSource(vs, 1, &vss, 0);
		glCompileShader(vs);
		glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
		if ( len > 0 ) nvprintf  ( "ERROR Shader2D vert: %s\n", buf );
		checkGL( "Compile vertex shader" );

		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		GLchar const * fss =
			"#version 420\n"
			"\n"		
			"uniform sampler2D fontTex;\n"
			"in vec3 position;\n"		
			"in vec4 color;\n"
			"in vec2 texcoord;\n"		
			"out vec4 outColor;\n"
			"\n"
			"void main()\n"
			"{\n"					
			"    vec4 fontclr = texture(fontTex, texcoord);\n"
			"    outColor = (color.w==-1) ? fontclr : color * vec4(1,1,1, fontclr.x); \n"		
			"}\n"
		;		

		

		glShaderSource(fs, 1, &fss, 0);
		glCompileShader(fs);
		glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
		if ( len > 0 ) nvprintf  ( "ERROR Shader2D frag: %s\n", buf );
		checkGL( "Compile fragment shader" );

		mSH[ SCOLOR ] = glCreateProgram();
		glAttachShader( mSH[ SCOLOR ], vs);
		glAttachShader( mSH[ SCOLOR ], fs);
		checkGL( "Attach program" );
		glLinkProgram( mSH[SCOLOR] );
		checkGL( "Link program" );
		glUseProgram( mSH[SCOLOR] );
		checkGL( "Use program" );

		mProj[SCOLOR] =		glGetProgramResourceIndex ( mSH[SCOLOR], GL_UNIFORM, "projMatrix" );	
		mModel[SCOLOR] =	glGetProgramResourceIndex ( mSH[SCOLOR], GL_UNIFORM, "modelMatrix" );	
		mView[SCOLOR] =		glGetProgramResourceIndex ( mSH[SCOLOR], GL_UNIFORM, "viewMatrix" );	
		mFont[SCOLOR] =		glGetProgramResourceIndex ( mSH[SCOLOR], GL_UNIFORM, "fontTex" );	
		checkGL( "Get Shader Matrices" );
	#endif
}

void nvDraw::CreateSInst ()
{

	// OpenGL - Create shaders
	char buf[16384];
	int len = 0;
	checkGL( "Start shaders" );

	// OpenGL 4.2 Core
	// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLchar const * vss =
		"#version 430\n"
		"#extension GL_ARB_texture_rectangle : enable\n"
		"#extension GL_ARB_explicit_attrib_location : enable\n"
		"#extension GL_ARB_separate_shader_objects : enable\n"
		"layout(location = 0) in vec3	vertex;		// geometry attributes\n"
		"layout(location = 1) in vec4	color;		\n"
		"layout(location = 2) in vec2	texcoord;	\n"
		"layout(location = 3) in vec3	instPos1;	// instance attributes\n"
		"layout(location = 4) in vec4	instClr;   \n"
		"layout(location = 5) in vec2	instUV;    \n"
		"layout(location = 6) in vec3	instPos2;  \n"
		"uniform mat4			viewMatrix;			\n"
		"uniform mat4			projMatrix;			\n"
		"uniform mat4			modelMatrix;		\n"
		"out vec4 vworldpos;		\n"
		"out vec4 vcolor;			\n"
		"flat out vec3 vtexcoord1;	\n"
		"flat out vec3 vtexcoord2;	\n"
		"flat out mat4 vxform;		\n"
		"void main() {\n"
		"  int inst = gl_InstanceID;	\n"
		"  vxform = mat4 ( instPos2.x-instPos1.x, 0, 0, 0, \n"
		"			0, instPos2.y-instPos1.y, 0, 0,   \n"
		"			0, 0, instPos2.z-instPos1.z, 0,   \n"
		"			instPos1.x, instPos1.y, instPos1.z, 1);	\n"
		"  vworldpos = vxform * vec4(vertex,1);\n"
		"  vcolor = instClr;\n"
	    "  gl_Position = projMatrix * viewMatrix * vworldpos;\n"
        "}\n"
	;
	glShaderSource(vs, 1, &vss, 0);
	glCompileShader(vs);
	glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) nvprintf  ( "ERROR ShaderInst vert: %s\n", buf );
	checkGL( "Compile vertex shader" );

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLchar const * fss =
		"#version 430\n"
		"#extension GL_ARB_explicit_attrib_location : enable \n"
		"#extension GL_ARB_separate_shader_objects : enable \n"
		"#extension GL_ARB_texture_rectangle : enable \n"
		"in vec4 vworldpos; \n"
		"in vec4 vcolor; \n"
		"flat in mat4 vxform; \n"
		"flat in vec3 vtexcoord1; \n"
		"flat in vec3 vtexcoord2; \n"
		"out vec4 outColor;\n"
		"void main () {\n"
		"  outColor = vcolor;\n"
		"}\n"
	;
	glShaderSource(fs, 1, &fss, 0);
	glCompileShader(fs);
	glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) nvprintf  ( "ERROR ShaderInst frag: %s\n", buf );
	checkGL( "Compile fragment shader" );

	mSH[SINST] = glCreateProgram();
	glAttachShader( mSH[SINST], vs);
	glAttachShader( mSH[SINST], fs);
	checkGL( "Attach program" );
	glLinkProgram( mSH[SINST] );
	checkGL( "Link program" );
	glUseProgram( mSH[SINST] );
	checkGL( "Use program" );

	mProj[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "projMatrix" );	
	mModel[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "modelMatrix" );	
	mView[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "viewMatrix" );		
	checkGL( "Get Shader Matrices" );	
}

void nvDraw::CreateS3D ()
{

	// OpenGL - Create shaders
	char buf[16384];
	int len = 0;
	checkGL( "Start shaders" );

	// OpenGL 4.2 Core
	// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLchar const * vss =
			"#version 420\n"
			"\n"
			"layout(location = 0) in vec3 inPosition;\n"
			"layout(location = 1) in vec4 inColor;\n"
			"layout(location = 2) in vec2 inTexCoord;\n"
			"layout(location = 3) in vec3 inNorm;\n"
			"out vec3 vpos;\n"		
			"out vec4 vcolor;\n"				
			"out vec2 vtexcoord;\n"
			"out vec3 vnorm;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"		
			"out gl_PerVertex {\n"
			"   vec4 gl_Position;\n"		
			"};\n"
			"\n"
			"void main()\n"
			"{\n"		
			"	 vpos = (modelMatrix * vec4(inPosition,1)).xyz;\n"
			"    vcolor = inColor;\n"
			"    vtexcoord = inTexCoord;\n"
			"	 vnorm = inNorm;\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(inPosition,1);\n"
			"}\n"
	;
	glShaderSource(vs, 1, &vss, 0);
	glCompileShader(vs);
	glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) nvprintf  ( "ERROR ShaderInst vert: %s\n", buf );
	checkGL( "Compile vertex shader" );

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLchar const * fss =
		"#version 430\n"		
		"in vec3		vpos; \n"
		"in vec4		vcolor; \n"		
		"in vec2		vtexcoord; \n"
		"in vec3		vnorm; \n"		
		"uniform vec3	lightpos; \n"
		"out vec4		outColor;\n"
		"void main () {\n"
		"   float d = clamp( dot ( vnorm, normalize(lightpos-vpos) ), 0, 1 ); \n"
		"   outColor = vec4(d, d, d, 1);\n"
		"}\n"
	;
	glShaderSource(fs, 1, &fss, 0);
	glCompileShader(fs);
	glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) nvprintf  ( "ERROR ShaderInst frag: %s\n", buf );
	checkGL( "Compile fragment shader" );

	mSH[S3D] = glCreateProgram();
	glAttachShader( mSH[S3D], vs);
	glAttachShader( mSH[S3D], fs);
	checkGL( "Attach program" );
	glLinkProgram( mSH[S3D] );
	checkGL( "Link program" );
	glUseProgram( mSH[S3D] );
	checkGL( "Use program" );

	mProj[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "projMatrix" );	
	mModel[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "modelMatrix" );	
	mView[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "viewMatrix" );	
	mLight[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "lightpos" );	
	checkGL( "Get Shader Matrices" );	
}

void nvDraw::drawGL ()
{
	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_TEXTURE_2D );	
	glEnable ( GL_BLEND );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	glBindVertexArray ( mVAO );		
	glUseProgram ( mSH[S3D] );	
	if ( m3D.size()==0 ) { nvprintf ( "ERROR: Must have used start3D before using drawGL\n" ); }
	nvSet* s = &m3D[ 0 ];	
	glProgramUniformMatrix4fv ( mSH[S3D], mProj[S3D],  1, GL_FALSE, s->proj );	
	glProgramUniformMatrix4fv ( mSH[S3D], mModel[S3D], 1, GL_FALSE, s->model ); 
	glProgramUniformMatrix4fv ( mSH[S3D], mView[S3D],  1, GL_FALSE, s->view );		
	glActiveTexture ( GL_TEXTURE0 );		
	glBindTexture ( GL_TEXTURE_2D, mWhiteImg.getTex() );
	checkGL ( "drawGL" );
}
void nvDraw::setLight (int s, float x1, float y1, float z1 )
{
	glProgramUniform3f ( mSH[s], mLight[s], x1, y1, z1 );
}


bool nvDraw::Initialize ( const char* fontName )
{
	CreateSColor ();
	CreateSInst ();
	CreateS3D ();

	if ( mVAO != 65535 ) {
		nvprintf ( "ERROR: init2D was already called.\n" );
		nverror ();
	}
	#ifdef USE_DX	
		// DirectX - Create model/view/proj buffers
		D3D11_BUFFER_DESC bd; 
		ZeroMemory( &bd, sizeof(bd) ); 
		bd.Usage = D3D11_USAGE_DEFAULT; 
		bd.ByteWidth = sizeof(MatrixBuffer); 
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; 
		bd.CPUAccessFlags = 0;
		HRESULT hr;
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[0] ), "CreateBuffer" );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[1] ), "CreateBuffer" );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[2] ), "CreateBuffer" );
	#else
		// OpenGL - Create VAO
		glGenVertexArrays ( 1, &mVAO );	
	#endif
	glBindVertexArray ( mVAO );

	memset ( &mBoxSet, 0, sizeof(nvSet) );
	mCurrSet = &mBoxSet;
	drawLine3D ( 0, 0, 0, 1, 0, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 0, 1, 0, 1, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 1, 0, 0, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 );
	drawLine3D ( 0, 1, 0, 1, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 );	
	drawLine3D ( 0, 0, 0, 0, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 0, 1, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 0, 1, 0, 1, 1, 1, 1, 1, 1 );
	UpdateVBOs ( mBoxSet );

	mWhiteImg.Create ( 8, 8, IMG_RGBA );
	mWhiteImg.Fill ( 1,1,1,1 );

	if ( !LoadFont ( fontName ) ) return false;
	
	return true;
}

void nvDraw::UpdateVBOs ( nvSet& s )
{	
	#ifdef USE_DX	

		for (int n=0; n < GRP_MAX; n++) {
			#ifdef DEBUG_UTIL
				nvprintf  ( "Draw::UpdateVBOs: %d of %d.\n", n, GRP_MAX );
			#endif
			if ( s.mNum[n] == 0 ) continue;

			if ( s.mVBO[n] != 0x0 ) s.mVBO[n]->Release ();

			D3D11_BUFFER_DESC bd; 
			ZeroMemory( &bd, sizeof(bd) ); 
			bd.Usage = D3D11_USAGE_DYNAMIC;
			bd.ByteWidth = s.mNum[n] * sizeof(nvVert); 
			bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bd.StructureByteStride = 0;				
			checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBO[n] ), "CreateBuffer(VBO)" );				
			
			// create index buffer 
			bd.ByteWidth = s.mNumI[n] * sizeof(uint);
			bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
			checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBOI[n] ), "CreateBuffer(VBO)" );					
			
			D3D11_MAPPED_SUBRESOURCE resrc;
			ZeroMemory( &resrc, sizeof(resrc) ); 
			checkHR( g_pContext->Map ( s.mVBO[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
			memcpy ( resrc.pData, s.mGeom[n], s.mNum[n] * sizeof(nvVert) );
			g_pContext->Unmap ( s.mVBO[n], 0 );
			
			checkHR( g_pContext->Map ( s.mVBOI[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
			memcpy ( resrc.pData, s.mIdx[n], s.mNumI[n] * sizeof(uint) );		
			g_pContext->Unmap ( s.mVBOI[n], 0 );
		} 
		
	#else		

		for (int n=0; n < GRP_MAX; n++ ) {
			if ( s.mNum[n] == 0 ) continue; 
			if ( s.mVBO[n] == 0 ) glGenBuffers ( 1, &s.mVBO[n] );		
			// bind buffer /w data
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[n] );			
			glBufferData ( GL_ARRAY_BUFFER, s.mNum[n] * sizeof(nvVert), s.mGeom[n], GL_STATIC_DRAW_ARB);	
			checkGL ( "update set buffer data" );
			// bind index buffer - not necessary in GL when using glDrawElements
			
			//-- debugging
			//if ( n == GRP_IMG ) 
			//	for (int j=0; j < s.mNum[n]; j++ ) nvprintf  ( "%d  %f,%f,%f\n", j, s.mGeom[n][j].x, s.mGeom[n][j].y, s.mGeom[n][j].z );
		}
	#endif
}

void nvDraw::drawSet2D ( nvSet& s )
{
	if ( s.zfactor == 1.0 ) 
		glDisable ( GL_DEPTH_TEST );		// don't preserve order
	else					
		glEnable ( GL_DEPTH_TEST );			// preserve order
	glEnable ( GL_TEXTURE_2D );	
		
	glProgramUniform1i ( mSH[SCOLOR], mFont[SCOLOR], 0 );		
	glActiveTexture ( GL_TEXTURE0 );		
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );	
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mProj[SCOLOR],  1, GL_FALSE, s.proj );	
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mModel[SCOLOR], 1, GL_FALSE, s.model ); 
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mView[SCOLOR],  1, GL_FALSE, s.view );	
	checkGL ( "matrices" );

	// lines 
	glBindTexture ( GL_TEXTURE_2D, mWhiteImg.getTex() );	
	GLsizei sz;
	if ( s.mVBO[GRP_LINES] !=0 && s.mNum[GRP_LINES] != 0 ) {
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_LINES ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNum[GRP_LINES];
		glDrawArrays ( GL_LINES, 0, sz );
	}
	checkGL ( "lines" );

	// triangles			
	glEnable ( GL_PRIMITIVE_RESTART );    		
	glPrimitiveRestartIndex ( IDX_NULL );	
	if ( s.mVBO[GRP_TRI] !=0 && s.mNum[GRP_TRI] != 0 ) {			
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRI ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNumI[GRP_TRI];
		glDrawElements ( GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRI] );		
	}
	checkGL ( "triangles" );

	// images
	// * Note: Must be drawn individually unless we use bindless
	int pos=0;
	nvImg* img;
	
	for (int n=0; n < s.mNum[GRP_IMG] / 4 ; n++ ) {
		img = s.mImgs[n];		
		glBindTexture ( GL_TEXTURE_2D, img->getTex());										
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_IMG ] );	
		glEnableVertexAttribArray ( localPos );
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 0) );
		
		glEnableVertexAttribArray ( localClr );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 12) );

		glEnableVertexAttribArray ( localUV );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 28) );

		glDrawArrays (GL_TRIANGLE_STRIP, 0, 4 );			
		pos += sizeof(nvVert)*4;
	}
	checkGL ( "images" );

	// text
	if ( s.mVBO[GRP_TRITEX] !=0 && s.mNum[GRP_TRITEX] != 0 ) {
		glBindTexture ( GL_TEXTURE_2D, mFontImg.getTex() );								
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRITEX ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNumI[GRP_TRITEX];
		glDrawElements ( GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRITEX] );
		checkGL ( "text" );
	}		
	glDisable ( GL_PRIMITIVE_RESTART );
		
	glDisable ( GL_TEXTURE_2D );		
}


void nvDraw::drawSet3D ( nvSet& s )
{
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mProj[SCOLOR],  1, GL_FALSE, s.proj );	
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mModel[SCOLOR], 1, GL_FALSE, s.model ); 
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mView[SCOLOR],  1, GL_FALSE, s.view );	
	GLsizei sz;

	// lines 	
	if ( s.mVBO[GRP_LINES] !=0 && s.mNum[GRP_LINES] != 0 ) {
		glBindTexture ( GL_TEXTURE_2D, mWhiteImg.getTex() );	
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_LINES ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNum[GRP_LINES];
		glDrawArrays ( GL_LINES, 0, sz );
		checkGL ( "draw3D lines" );
	}

	// box 3D (instanced)
	if ( s.mVBO[GRP_BOX] !=0 && s.mNum[GRP_BOX] != 0 ) {						
		glUseProgram ( mSH[SINST] );							
		glProgramUniformMatrix4fv ( mSH[SINST], mProj[SINST],  1, GL_FALSE, s.proj );	
		glProgramUniformMatrix4fv ( mSH[SINST], mModel[SINST], 1, GL_FALSE, s.model ); 
		glProgramUniformMatrix4fv ( mSH[SINST], mView[SINST],  1, GL_FALSE, s.view );
			
		glBindBuffer ( GL_ARRAY_BUFFER, mBoxSet.mVBO[GRP_LINES] );		// bind cube geometry			
		glEnableVertexAttribArray ( localPos );			
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );			
		glEnableVertexAttribArray ( localClr );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );			
		glEnableVertexAttribArray ( localUV );			
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );						
			
		// bind instances						
		// a single instance is two verticies wide, so use stride of sizeof(nvVert)*2
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[GRP_BOX] );				
		glEnableVertexAttribArray ( attrPos );
		glVertexAttribPointer( attrPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, 0 );
		glVertexAttribDivisor ( attrPos, 1 );
		glEnableVertexAttribArray ( attrClr );   
		glVertexAttribPointer( attrClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 12 );
		glVertexAttribDivisor ( attrClr, 1 );
		glEnableVertexAttribArray ( attrUV );
		glVertexAttribPointer( attrUV,   2, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 28 );
		glVertexAttribDivisor ( attrUV, 1 );
		glEnableVertexAttribArray ( attrPos2 );
		glVertexAttribPointer( attrPos2, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 36 );			
		glVertexAttribDivisor ( attrPos2, 1 );						

		sz = (GLsizei) mBoxSet.mNum[GRP_LINES];
		glDrawArraysInstanced ( GL_LINES, 0, sz, (GLsizei) s.mNum[GRP_BOX] );			
		checkGL ( "draw3D boxes" );
			
		glDisableVertexAttribArray ( attrPos );
		glDisableVertexAttribArray ( attrClr );
		glDisableVertexAttribArray ( attrUV );
		glDisableVertexAttribArray ( attrPos2 );
		glUseProgram ( mSH[SCOLOR] );				
	}	
}



void nvDraw::draw2D ()
{
	#ifdef USE_DX
		// Set 2D shader
		g_pContext->VSSetShader ( mVS, 0, 0);
		g_pContext->PSSetShader ( mPS, 0, 0);
		g_pContext->IASetInputLayout( mLO );		
		g_pContext->OMSetDepthStencilState( g_pDepthOffState, 1 );
	#else
		glEnable ( GL_BLEND );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable ( GL_TEXTURE_3D );
		glDisable ( GL_LIGHTING );
		glDisable ( GL_POLYGON_OFFSET_LINE );
		glDisable ( GL_CULL_FACE );
		glDisable ( GL_ALPHA_TEST );		
		glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

		glBindVertexArray ( mVAO );		
		glUseProgram ( mSH[SCOLOR] );		
	#endif
	checkGL ( "prep2D" );

	// Update VBOs for Dynamic Draw sets
	// (disable attrib arrays first)
	glDisableVertexAttribArray( 0 );
	glDisableVertexAttribArray( 1 );
	glDisableVertexAttribArray( 2 );
	glDisableVertexAttribArray( 3 );
	std::vector<nvSet>::iterator it;	
	it = mDynamic.begin();	
	for ( int n = 0; n < mDynNum; n++ ) 
		UpdateVBOs ( (*it++) );
	
	// Draw Sets	
	// (attrib arrays enabled)
	glEnableVertexAttribArray( localPos );
	glEnableVertexAttribArray( localClr );
	glEnableVertexAttribArray( localUV  );
	for ( it = mStatic.begin(); it != mStatic.end(); it++ ) 
		drawSet2D ( (*it) );	
	it = mDynamic.begin();	
	for ( int n = 0; n < mDynNum; n++ ) 
		drawSet2D ( (*it++) );	

	// Delete dynamic buffers	
	nvSet* s;
	for (int n=0; n < mDynamic.size(); n++ ) {		
		s = &mDynamic[n];
		for (int grp=0; grp < GRP_MAX; grp++) {
			if ( s->mGeom[grp] != 0x0 ) delete s->mGeom[grp]; 
			if ( s->mIdx[grp] != 0x0 )  delete s->mIdx[grp];	
			s->mNum[grp] = 0;	s->mMax[grp] = 0;	s->mGeom[grp] = 0;
			s->mNumI[grp] = 0;	s->mMaxI[grp] = 0;	s->mIdx[grp] = 0;
		}		
	}
	mDynNum = 0;	// reset first dynamic buffer (reuses VBOs)	

	mCurrZ = 0;
}

void nvDraw::draw3D ()
{
	std::vector<nvSet>::iterator it;

	glEnable ( GL_DEPTH_TEST );	
	glEnable ( GL_TEXTURE_2D );	
	glEnable ( GL_BLEND );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	glBindVertexArray ( mVAO );		
	glUseProgram ( mSH[SCOLOR] );
	glProgramUniform1i ( mSH[SCOLOR] , mFont[SCOLOR] , 0 );		
	glActiveTexture ( GL_TEXTURE0 );		
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );		
	checkGL ( "draw3D prep" );
	
	glDisableVertexAttribArray( 0 );
	glDisableVertexAttribArray( 1 );
	glDisableVertexAttribArray( 2 );
	glDisableVertexAttribArray( 3 );

	it = m3D.begin();
	for ( int n = 0; n < m3DNum; n++ ) 
		UpdateVBOs ( (*it++) );	

	glEnableVertexAttribArray( localPos );
	glEnableVertexAttribArray( localClr );
	glEnableVertexAttribArray( localUV  );	

	it = m3D.begin();
	for ( int n = 0; n < m3DNum; n++ ) 
		drawSet3D ( (*it++) );
		
	m3DNum = 0;			// reset dynamic buffers

	glUseProgram ( 0 );
	glBindVertexArray ( 0 );
	checkGL ( "draw3D done" );
}

bool nvDraw::LoadFont ( const char * fontName )
{
    if (!fontName) return false;

    char fname[200], fpath[1024];
    sprintf (fname, "%s.tga", fontName);
	if ( !getFileLocation ( fname, fpath ) ) {
		nvprintf ( "ERROR: Cannot find %s.\n", fname );		
	}
	
	mFontImg.LoadTga ( fpath );

	// change extension from .tga to .bin
	int l = (int) strlen(fpath);
	fpath[l-3] = 'b';	fpath[l-2] = 'i'; fpath[l-1] = 'n';	fpath[l-0] = '\0';
    FILE *fd = fopen( fpath, "rb" );
    if ( !fd ) return false;

    int r = (int)fread(&mGlyphInfos, 1, sizeof(FileHeader), fd);
    fclose(fd);

	return true;
}

//--------------------------------------- 2D GUIs

nvGui::nvGui ()
{
	g_GuiCallback = 0;
	mActiveGui = -1;
}

int nvGui::AddGui ( float x, float y, float w, float h, const char* name, int gtype, int dtype, void* data, float vmin, float vmax  )
{
	Gui g;

	g.name = name;
	g.x = x; g.y = y;
	g.w = w; g.h = h;
	g.gtype = gtype;
	g.dtype = dtype;
	g.data = data;
	g.vmin = vmin;
	g.vmax = vmax;
	g.items.clear ();
	g.backclr.Set ( 0.5f, 0.5f, 0.5f, 0.7f );
	if ( gtype == GUI_ICON || gtype == GUI_TOOLBAR )
		g.backclr.Set ( 0, 0, 0, 0 );
	
	mGui.push_back ( g );
	return (int) mGui.size()-1;
}



int nvGui::AddItem ( char* name, char* imgname )
{
	int g = (int) mGui.size()-1;
	mGui[g].items.push_back ( name );
	nvImg* img = 0x0;
	
	if ( imgname != 0x0 ) {
		char fpath[1024];
		if ( getFileLocation ( imgname, fpath ) ) {
			img = new nvImg;
			img->LoadPng ( fpath );
			img->FlipY ();			
		}		
	}
	mGui[g].imgs.push_back ( img );	

	return g;
}
void nvGui::SetBackclr ( float r, float g, float b, float a )
{
	int gi = (int) mGui.size()-1;
	mGui[gi].backclr = Vector4DF(r,g,b, a);
}

std::string nvGui::getItemName ( int g, int v )
{
	return mGui[g].items[v];
}

bool nvGui::guiChanged ( int n )
{
	if ( n < 0 || n >= mGui.size() ) return false;
	if ( mGui[n].changed ) {
		mGui[n].changed = false;
		return true;
	}
	return false;
}
void nvGui::Clear ()
{	
	// delete images
	for (int n = 0; n < mGui.size(); n++) {
		for (int j = 0; j < mGui[n].imgs.size(); j++) 
			delete mGui[n].imgs[j];			
		mGui[n].imgs.clear();
	}
	
	// clear
	mGui.clear ();
}
void nvGui::Draw ( nvImg* chrome )
{
	char buf[1024];
	float x1, y1, x2, y2, frac, dx, x3;
	float tx, ty;
	bool bval;

	start2D ();

	Vector4DF	tc ( 1, 1, 1, 1);		// text color
	Vector3DF	toff ( 5, 15, 0 );		// text offset
	
	for (int n=0; n < mGui.size(); n++ ) {
		
		x1 = mGui[n].x;	y1 = mGui[n].y;
		x2 = x1 + mGui[n].w; y2 = y1 + mGui[n].h;	
		tx = x1 + toff.x; ty = y1 + toff.y;
		x3 = x2 - mGui[n].w/2;

		if ( chrome != 0x0 ) drawImg ( chrome, x1, y1, x2, y2, 1, 1, 1, 1 );		
		else				 drawFill ( x1, y1, x2, y2, mGui[n].backclr.x, mGui[n].backclr.y, mGui[n].backclr.z, mGui[n].backclr.w );
		
		switch ( mGui[n].gtype ) {
		case GUI_PRINT: {
			
			#ifdef DEBUG_UTIL
				nvprintf  ( "Gui::Draw: Textbox. %d of %d\n", n, mGui.size() );
			#endif
			switch ( mGui[n].dtype ) {
			case GUI_STR:	sprintf ( buf, "%s", ((std::string*) mGui[n].data)->c_str() );	break;
			case GUI_INT:	sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_VEC3:	{
				Vector3DF* val = (Vector3DF*) mGui[n].data;
				sprintf ( buf, "%4.0f x %4.0f x %4.0f", val->x, val->y, val->z );
				} break;
			case GUI_FLOAT: sprintf ( buf, "%.5f", *(float*) mGui[n].data );	break;
			case GUI_BOOL:  if (*(bool*) mGui[n].data) sprintf (buf, "on" ); else sprintf(buf,"off");	break;
			};
			dx = getTextX (buf);
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tc.w );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tc.w );	
			} break;
		case GUI_SLIDER: {				
			#ifdef DEBUG_UTIL
				nvprintf  ( "Gui::Draw: Slider. %d of %d\n", n, mGui.size() );
			#endif			
			switch ( mGui[n].dtype ) {
			case GUI_INT:	frac = (float(*(int  *) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_FLOAT: frac = (     (*(float*) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%.3f", *(float*) mGui[n].data );	break;
			};			
			drawFill ( x3, y1+2, x3+frac*(x2-x3), y2-2, .6f, 1.0f, .8f, 1.0f );		
			dx = getTextX (buf);
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tc.w );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tc.w );		
			} break;
		case GUI_CHECK: {
			#ifdef DEBUG_UTIL
				nvprintf  ( "Gui::Draw: Checkbox. %d of %d\n", n, mGui.size() );
			#endif			
			switch ( mGui[n].dtype ) {
			case GUI_INT:		bval = (*(int*) mGui[n].data) == 0 ? false : true;	break;
			case GUI_FLOAT:		bval = (*(float*) mGui[n].data) == 0.0 ? false : true;	break;
			case GUI_BOOL:		bval = *(bool*) mGui[n].data;	break;
			};
			if ( bval ) {
				drawText ( x2-40, ty, "On", tc.x, tc.y, tc.z, tc.w );				
			} else {
				drawText ( x2-40, ty, "Off", tc.x, tc.y, tc.z, tc.w);
			}
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tc.w );		
			} break;
		case GUI_COMBO: {
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tc.w);
			
			int val = *(int*) mGui[n].data;
			if ( val >=0 && val < mGui[n].items.size() ) {
				sprintf ( buf, "%s", mGui[n].items[val].c_str() );
			} else {
				sprintf ( buf, "" );
			}
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tc.w );
			} break;
		case GUI_TOOLBAR: {
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			int iw = mGui[n].imgs[0]->getWidth();
			int ih = mGui[n].imgs[0]->getHeight();
			float ix, iy;
			char msg[1024];

			int val = *(int*) mGui[n].data;
			ix = x1 + val*(iw + 25); iy = y1;
			drawFill ( ix, iy, ix+iw, iy+ih, 1.0, 1.0, 1.0, 0.25 );

			for (int j=0; j < mGui[n].items.size(); j++ ) {      // buttons
				ix = x1 + j*(iw + 25); iy = y1;
				drawImg ( mGui[n].imgs[j], ix, iy, ix+iw, iy+ih, 1, 1, 1, 1 );
				strcpy ( msg, mGui[n].items[j].c_str() );
				drawText ( ix, y2, msg, tc.x, tc.y, tc.z, tc.w );				
			}
			}; break;

		case GUI_ICON:	{
			drawImg ( mGui[n].imgs[0], x1, y1, x2, y2, 1, 1, 1, 1 );
			}; break;

		}
	}
	end2D ();

}

bool nvGui::MouseUp ( float x, float y )
{
	int ag = mActiveGui; 
	mActiveGui = -1;
	return (ag!=-1);
}

bool nvGui::MouseDown ( float x, float y )
{
	// GUI down - Check if GUI is hit
	float xoff = 150;
	float x1, y1, x2, x3, y2;
	for (int n=0; n < mGui.size(); n++ ) {
		x1 = mGui[n].x;			y1 = mGui[n].y;
		x2 = x1 + mGui[n].w;	y2 = y1 + mGui[n].h;		
		x3 = x2 - xoff;

		switch ( mGui[n].gtype ) {
		case GUI_SLIDER:
			if ( x > x3 && x < x2 && y > y1 && y < y2) {
				mActiveGui = n;	  	// set active gui								
				return true;
			}
			break;
		case GUI_CHECK: 
			if ( x > x1 && x < x2 && y > y1 && y < y2 ) {
				mActiveGui = -1;
				mGui[ n ].changed = true;
				int val;
				switch ( mGui[ n ].dtype ) {
				case GUI_INT:	val = ( (*(int*) mGui[n].data) == 0 ) ? 1 : 0;			*(int*) mGui[n].data = (int) val;		break;
				case GUI_FLOAT:	val = ( (*(float*) mGui[n].data) == 0.0 ) ? 1 : 0;		*(float*) mGui[n].data = (float) val;	break;
				case GUI_BOOL:	val = ( (*(bool*) mGui[n].data) == false ) ? 1 : 0;		*(bool*) mGui[n].data = (val==0) ? false : true;		break;
				};
				if ( g_GuiCallback ) g_GuiCallback( n, (float) val );
				return true;
			}
			break;
		case GUI_COMBO:
			if ( x > x2-xoff && x < x2 && y > y1 && y < y2 ) {
				mActiveGui = n;
				mGui [ n ].changed = true;								// combe box has changed
				int val = *(int*) mGui[n].data;							// get combo id
				val = (val >= mGui[n].items.size()-1 ) ? 0 : val+1;		// increment value
				*(int*) mGui[n].data = val;
				if ( g_GuiCallback ) g_GuiCallback( n, (float) val );
			} 
			break;
		case GUI_TOOLBAR: {
			int iw = mGui[n].imgs[0]->getWidth();
			int ih = mGui[n].imgs[0]->getHeight();
			float ix, iy;

			for (int j=0; j < mGui[n].items.size(); j++ ) {      // buttons
				ix = x1 + j*(iw + 25); iy = y1;
				if ( x > ix && y > iy && x < ix+iw && y < iy+ih ) {
					mActiveGui = n;
					mGui [ n ].changed = true;
					*(int*) mGui[n].data = j;
					if ( g_GuiCallback ) g_GuiCallback( n, (float) j );
				}
			}
			} break;
		default: break;


			
		};
	}
	mActiveGui = -1;
	return false;
}

bool nvGui::MouseDrag ( float x, float y )
{
	// GUI drag - Adjust value of hit gui
	float x1, y1, x2, x3, y2, val;
	float xoff = 150;
	if ( mActiveGui != -1 ) {
		x1 = mGui[ mActiveGui].x;			y1 = mGui[mActiveGui ].y;
		x2 = x1 + mGui[ mActiveGui ].w;	y2 = y1 + mGui[mActiveGui ].h;
		x3 = x2 - xoff;
		if ( x <= x3 ) {
			mGui[ mActiveGui ].changed = true;			
			val = mGui[ mActiveGui ].vmin;			
			if ( mGui[ mActiveGui ].dtype == GUI_INT ) val = (float) int(val); 
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
		if ( x >= x2 ) {
			mGui[ mActiveGui ].changed = true;
			val = mGui[ mActiveGui ].vmax;		
			if ( mGui[ mActiveGui ].dtype == GUI_INT ) val = (float) int(val); 
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
		if ( x > x3 && x < x2 ) {
			mGui[ mActiveGui ].changed = true;
			switch ( mGui[ mActiveGui ].dtype ) {
			case GUI_INT:	val = (float) int( mGui[ mActiveGui ].vmin +   (x-x3)*mGui[ mActiveGui ].vmax / (x2-x3) );	 break;
			case GUI_FLOAT:	val = (float) (mGui[ mActiveGui ].vmin + (x-x3)*mGui[ mActiveGui ].vmax / (x2-x3));	 break;						
			};
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
	}
	return false;
}


bool readword ( char *line, char *word, char delim )
{
    int max_size = 200;
    char *buf_pos;
    char *start_pos;

    // read past spaces/tabs, or until end of line/string
    for (buf_pos=line; (*buf_pos==' ' || *buf_pos=='\t') && *buf_pos!='\n' && *buf_pos!='\0';)
        buf_pos++;

    // if end of line/string found, then no words found, return null
    if (*buf_pos=='\n' || *buf_pos=='\0') {*word = '\0'; return false;}

    // mark beginning of word, read until end of word
    for (start_pos = buf_pos; *buf_pos != delim && *buf_pos!='\t' && *buf_pos!='\n' && *buf_pos!='\0';)
        buf_pos++;

    if (*buf_pos=='\n' || *buf_pos=='\0') {    // buf_pos now points to the end of buffer
        //strcpy_s (word, max_size, start_pos);            // copy word to output string
        strncpy (word, start_pos, max_size);
        if ( *buf_pos=='\n') *(word + strlen(word)-1) = '\0';
        *line = '\0';                        // clear input buffer
    } else {
                                            // buf_pos now points to the delimiter after word
        *buf_pos++ = '\0';                    // replace delimiter with end-of-word marker
        //strcpy_s (word, max_size, start_pos);
        strncpy (word, start_pos, buf_pos-line );    // copy word(s) string to output string
                                            // move start_pos to beginning of entire buffer
        strcpy ( start_pos, buf_pos );        // copy remainder of buffer to beginning of buffer
    }
    return true;                        // return word(s) copied
}



/*void save_png ( char* fname, unsigned char* img, int w, int h )
{
	unsigned error = lodepng::encode ( "test.png", img, w, h );	  
	if (error) printf ( "png write error: %s\n", lodepng_error_text(error) );
}*/

nvImg::nvImg ()
{
	mXres = 0;
	mYres = 0;
	mData = 0;
	mTex = UINT_NULL;
}

void nvImg::Create ( int x, int y, int fmt )
{
	mXres = x;
	mYres = y;
	mSize = mXres * mYres;
	mFmt = fmt;

	switch ( mFmt ) {
	case IMG_RGB:		mSize *= 3;	break;
	case IMG_RGBA:		mSize *= 4; break;
	case IMG_GREY16:	mSize *= 2; break;	
	}

    if ( mData != 0x0 ) free ( mData );
    mData = (unsigned char*) malloc ( mSize );
	 
	memset ( mData, 0, mSize );
    
	UpdateTex();
}

void nvImg::Fill ( float r, float g, float b, float a )
{
	unsigned char* pix = mData;
	for (int n=0; n < mXres*mYres; n++ ) {
	  *pix++ = (unsigned char) (r*255.0f); 
	  *pix++ = (unsigned char) (g*255.0f); 
	  *pix++ = (unsigned char) (b*255.0f); 
	  *pix++ = (unsigned char) (a*255.0f);
	}
	UpdateTex ();
}
void nvImg::FlipY ()
{
	int pitch = mSize / mYres;
	unsigned char* buf = (unsigned char*) malloc ( pitch );
	for (int y=0; y < mYres/2; y++ ) {
		memcpy ( buf, mData + (y*pitch), pitch );		
		memcpy ( mData + (y*pitch), mData + ((mYres-y-1)*pitch), pitch );		
		memcpy ( mData + ((mYres-y-1)*pitch), buf, pitch );
	}
	UpdateTex ();
}


bool nvImg::LoadPng ( char* fname, bool bGrey )
{
	std::vector< unsigned char > out;
	unsigned int w, h; 

	unsigned error = lodepng::decode ( out, w, h, fname, (bGrey ? LCT_GREY : LCT_RGBA), (bGrey ? 16 : 8) );
	if (error) {
		nvprintf  ( "png read error: %s\n", lodepng_error_text(error) );
		return false;
	}	
	Create ( w, h, (bGrey ? IMG_GREY16 : IMG_RGBA) );
	int stride = mSize / mYres;

	for (int y=0; y < mYres; y++ ) 
		memcpy ( mData + y*stride, &out[ y*stride ], stride );

	UpdateTex ();

	return true;
}

void nvImg::SavePng ( char* fname )
{
	nvprintf  ( "Saving PNG: %s\n", fname );
	save_png ( fname, mData, mXres, mYres, 4 );
}

bool nvImg::LoadTga ( char* fname )
{
	nvprintf  ( "Reading TGA: %s\n", fname );
	TGA* fontTGA = new TGA;
    TGA::TGAError err = fontTGA->load(fname);
    if (err != TGA::TGA_NO_ERROR) {
		delete fontTGA;
		return false;  
	}
	 
	mXres = fontTGA->m_nImageWidth;
	mYres = fontTGA->m_nImageHeight;
	mSize = mXres * mYres;
	 
	switch ( fontTGA->m_texFormat ) {
	case TGA::RGB:		mFmt = IMG_RGB;		mSize *= 3;	break;
	case TGA::RGBA:		mFmt = IMG_RGBA;	mSize *= 4; break;
	case TGA::ALPHA:	mFmt = IMG_GREY16;	mSize *= 2;	break;
	case -1:
		delete fontTGA;
		return false;
	}

    if ( mData != 0x0 ) free ( mData );
    mData = (unsigned char*) malloc ( mSize );
	 
	memcpy ( mData, fontTGA->m_nImageData, mSize );
    
	UpdateTex();

	delete fontTGA;

	return true;
}


void nvImg::BindTex ()
{
	#ifdef USE_DX
		ID3D11ShaderResourceView* vlist[1];
		vlist[0] = mTexIV;
		g_pContext->PSSetShaderResources( 0, 1, vlist );
		g_pContext->PSSetSamplers ( 0, 1, &g_pSamplerState );
	#else	
		glBindTexture ( GL_TEXTURE_2D, mTex );
	#endif
}


void nvImg::UpdateTex ()
{
	#ifdef USE_DX

		unsigned char* fixed_data = mData;

		DXGI_FORMAT fmt;		
		int size;
		switch ( mFmt ) {
		case IMG_RGB: {			
			fmt = DXGI_FORMAT_R8G8B8A8_UNORM;	
			size = 4;
			fixed_data = (unsigned char*) malloc ( mXres*mYres*size );
			unsigned char* dest = fixed_data;
			unsigned char* src = mData;
			for (int y=0; y < mYres; y++ ) {
				for (int x=0; x < mXres; x++ ) {
					*dest++ = *(src+2);
					*dest++ = *(src+1);
					*dest++ = *(src);
					*dest++ = 255;
					src+=3;
				}
			}
			
			} break;  // !!!! RGB removed in DX11
		case IMG_RGBA:	fmt = DXGI_FORMAT_R8G8B8A8_UNORM;	size = 4; break;
		case IMG_LUM:	fmt = DXGI_FORMAT_R8_UNORM;			size = 1; break;
		}

		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory ( &desc, sizeof(desc) );		
		desc.Width = mXres;
		desc.Height = mYres;
		desc.MipLevels = desc.ArraySize = 1;
		desc.Format = fmt;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.MiscFlags = 0;		
		g_pDevice->CreateTexture2D ( &desc, 0, &mTex );

		D3D11_MAPPED_SUBRESOURCE resrc;
		ZeroMemory( &resrc, sizeof(resrc) ); 
		checkHR( g_pContext->Map ( mTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
		memcpy ( resrc.pData, fixed_data, mXres*mYres * size );
		g_pContext->Unmap ( mTex, 0 );
		
		D3D11_SHADER_RESOURCE_VIEW_DESC view_desc;		
		ZeroMemory ( &view_desc, sizeof(view_desc) );		
		view_desc.Format = desc.Format;
		view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		view_desc.Texture2D.MipLevels = 1;
		view_desc.Texture2D.MostDetailedMip = 0;
		g_pDevice->CreateShaderResourceView ( mTex, &view_desc, &mTexIV ); 

		if ( mFmt == IMG_RGB ) {
			free ( fixed_data );
		}

	#else
		if ( mTex != UINT_NULL ) 
			glDeleteTextures ( 1, (GLuint*) &mTex );
	
		//nvprintf  ( " Updating Texture %d x %d\n", mXres, mYres );
		glGenTextures ( 1, (GLuint*)&mTex );
		glBindTexture ( GL_TEXTURE_2D, mTex );
		checkGL ( "nvImg::UpdateTex" );
		glPixelStorei ( GL_PACK_ALIGNMENT, 1 );
		glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		
		GLenum fmt;
		int size;
		switch ( mFmt ) {
		case IMG_RGB:	fmt = GL_RGB; size = 3;			break;
		case IMG_RGBA:	fmt = GL_RGBA; size = 4;		break;
		case IMG_GREY16: fmt = GL_LUMINANCE; size = 2;	break;
		}

		glTexImage2D ( GL_TEXTURE_2D, 0, fmt, mXres, mYres, 0, fmt, GL_UNSIGNED_BYTE, mData );
	#endif
}
nvMesh::nvMesh ()
{
	mVBO.clear();
}

void nvMesh::Clear ()
{
	mVertices.clear ();
	mFaceVN.clear ();
	mNumFaces = 0;
	mNumSides = 0;
}

void nvMesh::ComputeNormals ()
{    
	int v1, v2, v3;
	Vector3DF p1, p2, p3;
	Vector3DF norm, side;

    // Clear vertex normals
    for (int n=0; n < mVertices.size(); n++) {
		mVertices[n].nx = 0;
		mVertices[n].ny = 0;
		mVertices[n].nz = 0;			
    }

    // Compute normals of all faces
    int n=0;
    for (int f = 0; f < mNumFaces; f++ ) {
		v1 = mFaceVN[f*3]; v2 = mFaceVN[f*3+1]; v3 = mFaceVN[f*3+2]; 
		p1.Set ( mVertices[v1].x, mVertices[v1].y, mVertices[v1].z );
		p2.Set ( mVertices[v2].x, mVertices[v2].y, mVertices[v2].z );
		p3.Set ( mVertices[v3].x, mVertices[v3].y, mVertices[v3].z );        
        norm = p2; norm -= p1; norm.Normalize ();
        side = p3; side -= p1; side.Normalize ();
        norm.Cross ( side );
		mVertices[v1].nx += norm.x; mVertices[v1].ny += norm.y; mVertices[v1].nz += norm.z; 
		mVertices[v2].nx += norm.x; mVertices[v2].ny += norm.y; mVertices[v2].nz += norm.z; 
		mVertices[v3].nx += norm.x; mVertices[v3].ny += norm.y; mVertices[v3].nz += norm.z; 
    }

    // Normalize vertex normals
    Vector3DF vec;
    for (int n=0; n < mVertices.size(); n++) {
		p1.Set ( mVertices[n].nx, mVertices[n].ny, mVertices[n].nz );
		p1.Normalize ();
		mVertices[n].nx = p1.x; mVertices[n].ny = p1.y; mVertices[n].nz = p1.z;
	}
}

void nvMesh::AddPlyElement ( char typ, int n )
{
    nvprintf  ( "  Element: %d, %d\n", typ, n );
    PlyElement* p = new PlyElement;
    if ( p == 0x0 ) { nvprintf  ( "ERROR: Unable to allocate PLY element.\n" ); }
    p->num = n;
    p->type = typ;
    p->prop_list.clear ();
    m_PlyCurrElem = (int) m_Ply.size();
    m_Ply.push_back ( p );
}

void nvMesh::AddPlyProperty ( char typ, std::string name )
{
    nvprintf  ( "  Property: %d, %s\n", typ, name.c_str() );
    PlyProperty p;
    p.name = name;
    p.type = typ;
    m_Ply [ m_PlyCurrElem ]->prop_list.push_back ( p );
}
int nvMesh::FindPlyElem ( char typ )
{
    for (int n=0; n < (int) m_Ply.size(); n++) {
        if ( m_Ply[n]->type == typ ) return n;
    }
    return -1;
}

int nvMesh::FindPlyProp ( int elem, std::string name )
{
    for (int n=0; n < (int) m_Ply[elem]->prop_list.size(); n++) {
        if ( m_Ply[elem]->prop_list[n].name.compare ( name)==0 )
            return n;
    }
    return -1;
}

bool nvMesh::LoadPly ( char* fname, float scal )
{
    FILE* fp;

    int m_PlyCnt;
    float m_PlyData[40];
    char buf[1000];
    char bword[200];
    std::string word;
    int vnum, fnum, elem, cnt;
    char typ;

    fp = fopen ( fname, "rt" );
    if ( fp == 0x0 ) { nvprintf  ( "ERROR: Could not find mesh file: %s\n", fname ); }

    // Read header
    fgets ( buf, 1000, fp );
    readword ( buf, bword, ' ' ); word = bword;
    if ( word.compare("ply" )!=0 ) {
		nvprintf  ( "ERROR: Not a ply file. %s\n", fname );        
    }

    m_Ply.clear ();

    nvprintf  ( "Reading PLY mesh: %s.\n", fname );
    while ( feof( fp ) == 0 ) {
        fgets ( buf, 1000, fp );
        readword ( buf, bword, ' ' );
        word = bword;
        if ( word.compare("comment" )!=0 ) {
            if ( word.compare("end_header")==0 ) break;
            if ( word.compare("property")==0 ) {
                readword ( buf, bword, ' ' );
                word = bword;
                if ( word.compare("float")==0 )		typ = PLY_FLOAT;
                if ( word.compare("float16")==0 )	typ = PLY_FLOAT;
                if ( word.compare("float32")==0 )	typ = PLY_FLOAT;
                if ( word.compare("int8")==0 )		typ = PLY_INT;
                if ( word.compare("uint8")==0 )		typ = PLY_UINT;
                if ( word.compare("list")==0) {
                    typ = PLY_LIST;
                    readword ( buf, bword, ' ' );
                    readword ( buf, bword, ' ' );
                }
                readword ( buf, bword, ' ' );
                word = bword;
                AddPlyProperty ( typ, word );
            }
            if ( word.compare("element" )==0 ) {
                readword ( buf, bword, ' ' );    word = bword;
                if ( word.compare("vertex")==0 ) {
                    readword ( buf, bword, ' ' );
                    vnum = atoi ( bword );
                    nvprintf  ( "  Verts: %d\n", vnum );
                    AddPlyElement ( PLY_VERTS, vnum );
                }
                if ( word.compare("face")==0 ) {
                    readword ( buf, bword, ' ' );
                    fnum = atoi ( bword );
                    nvprintf  ( "  Faces: %d\n", fnum );
                    AddPlyElement ( PLY_FACES, fnum );
                }
            }
        }
    }

    // Read data
    int xi, yi, zi, ui, vi;
    nvprintf  ( "Reading verts..\n" );
    elem = FindPlyElem ( PLY_VERTS );
    xi = FindPlyProp ( elem, "x" );
    yi = FindPlyProp ( elem, "y" );
    zi = FindPlyProp ( elem, "z" );
    ui = FindPlyProp ( elem, "s" );
    vi = FindPlyProp ( elem, "t" );
    if ( elem == -1 || xi == -1 || yi == -1 || zi == -1 ) {
        nvprintf  ( "ERROR: Vertex data not found.\n" );
    }

    xref vert;
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            readword ( buf, bword, ' ' );
            m_PlyData[ j ] = (float) atof ( bword );
        }
        vert = AddVert ( m_PlyData[xi]*scal, m_PlyData[yi]*scal, m_PlyData[zi]*scal, m_PlyData[ui], m_PlyData[vi], 0 );
    }

    nvprintf  ( "Reading faces..\n" );
    elem = FindPlyElem ( PLY_FACES );
    xi = FindPlyProp ( elem, "vertex_indices" );
    if ( elem == -1 || xi == -1 ) {
        nvprintf  ( "ERROR: Face data not found.\n" );
    }
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        m_PlyCnt = 0;
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            if ( m_Ply[elem]->prop_list[j].type == PLY_LIST ) {
                readword ( buf, bword, ' ' );
                cnt = atoi ( bword );
                m_PlyData[ m_PlyCnt++ ] = (float) cnt;
                for (int c =0; c < cnt; c++) {
                    readword ( buf, bword, ' ' );
                    m_PlyData[ m_PlyCnt++ ] = (float) atof ( bword );
                }
            } else {
                readword ( buf, bword, ' ' );
                m_PlyData[ m_PlyCnt++ ] = (float) atof ( bword );
            }
        }
        if ( m_PlyData[xi] == 3 ) {
            //debug.Printf ( "    Face: %d, %d, %d\n", (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3] );
            AddFace ( (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3] );
        }

        if ( m_PlyData[xi] == 4 ) {
            //debug.Printf ( "    Face: %d, %d, %d, %d\n", (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3], (int) m_PlyData[xi+4]);
           // AddFace ( (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3], (int) m_PlyData[xi+4] );
        }
    }
    for (int n=0; n < (int) m_Ply.size(); n++) {
        delete m_Ply[n];
    }
    m_Ply.clear ();
    m_PlyCurrElem = 0;

	nvprintf  ( "Computing normals.\n");
	ComputeNormals ();
	nvprintf  ( "Updating VBOs.\n");
	UpdateVBO( true );

	return 1;
}

int nvMesh::AddVert ( float x, float y, float z, float tx, float ty, float tz )
{
	Vertex v;	
	v.x = x; v.y = y; v.z = z;
	v.nx = v.x; v.ny = v.y; v.nz = v.z;
	float d = v.nx*v.nx+v.ny*v.ny+v.nz*v.nz;
	if ( d > 0 ) { d = sqrt(d); v.nx /= d; v.ny /= d; v.nz /= d; }
	
	v.tx = tx; v.ty = ty; v.tz = tz;
	
	mVertices.push_back ( v );
	return (int) mVertices.size()-1;
}
void nvMesh::SetVert ( int n, float x, float y, float z, float tx, float ty, float tz )
{
	mVertices[n].x = x;
	mVertices[n].y = y;
	mVertices[n].z = z;
	mVertices[n].tx = tx;
	mVertices[n].ty = ty;
	mVertices[n].tz = tz;
}
void nvMesh::SetNormal ( int n, float x, float y, float z )
{
	mVertices[n].nx = x;
	mVertices[n].ny = y;
	mVertices[n].nz = z;
}

int  nvMesh::AddFace ( int v0, int v1, int v2 )
{
	mFaceVN.push_back ( v0 );
	mFaceVN.push_back ( v1 );
	mFaceVN.push_back ( v2 );
	mNumFaces++;
	mNumSides = 3;
	return mNumFaces-1;
}
int nvMesh::AddFace4 ( int v0, int v1, int v2, int v3 )
{
	mFaceVN.push_back ( v0 );
	mFaceVN.push_back ( v1 );
	mFaceVN.push_back ( v2 );
	mFaceVN.push_back ( v3 );
	mNumFaces++;
	mNumSides = 4;
	return mNumFaces-1;
}

void nvMesh::UpdateVBO ( bool rebuild, int cnt )
{
	int numv = (int) mVertices.size();
	int numf = mNumFaces;

	#ifdef USE_DX
		if ( rebuild ) {
			#ifdef DEBUG_UTIL
				nvprintf  ( "nvMesh: UpdateVBO (rebuild)\n" );
			#endif
			if ( mVBO.size() == 0 ) {
				mVBO.push_back ( 0 );		// vertices
				mVBO.push_back ( 0 );		// faces	
			} else {
				mVBO[0]->Release ();
				mVBO[1]->Release ();
			}
			D3D11_BUFFER_DESC bd; 
			ZeroMemory( &bd, sizeof(bd) ); 
			bd.Usage = D3D11_USAGE_DYNAMIC; 
			bd.ByteWidth = numv * sizeof(Vertex); 
			bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bd.StructureByteStride = 0;

			D3D11_SUBRESOURCE_DATA InitData; 
			ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mVertices[0].x;			
			checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[0] ), "CreateBuffer(VBO)" );				
			
			bd.ByteWidth = numf * mNumSides * sizeof(unsigned int);
			bd.BindFlags = D3D11_BIND_INDEX_BUFFER; 
			bd.StructureByteStride = 0; //sizeof(unsigned int);			

			ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mFaceVN[0];
			checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[1] ), "CreateBuffer(VBO)"  );			

		} else {
			
			D3D11_MAPPED_SUBRESOURCE resrc;
			ZeroMemory( &resrc, sizeof(resrc) ); 
			checkHR( g_pContext->Map ( mVBO[0], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );

			#ifdef DEBUG_UTIL
				nvprintf  ( "nvMesh: map\n" );
			#endif
			for (int n=0; n < cnt; n++ ) {				
				memcpy ( resrc.pData, &mVertices[0].x, numv * sizeof(Vertex) );
			}
			g_pContext->Unmap ( mVBO[0], 0 );
			#ifdef DEBUG_UTIL
				nvprintf  ( "nvMesh: unmap\n" );
			#endif
		}

	#else		
		if ( mVBO.size()==0 ) {
			mVBO.push_back ( -1 );		// vertex buffer
			mVBO.push_back ( -1 );		// face buffer
		} else {
			glDeleteBuffers ( 1, &mVBO[0] );
			glDeleteBuffers ( 1, &mVBO[1] );
		}
		
		glGenBuffers ( 1, &mVBO[0] );
		glGenBuffers ( 1, &mVBO[1] );
		glGenVertexArrays ( 1, &mVAO );
		glBindVertexArray ( mVAO );
		glBindBuffer ( GL_ARRAY_BUFFER, mVBO[0] );
		glBufferData ( GL_ARRAY_BUFFER, mVertices.size() * sizeof(Vertex), &mVertices[0].x, GL_STATIC_DRAW_ARB);		
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localNorm, 3, GL_FLOAT, false, sizeof(Vertex), (void*) 12 );		
		glVertexAttribPointer ( localUV, 3, GL_FLOAT, false, sizeof(Vertex), (void*) 24 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );		
		glBufferData ( GL_ELEMENT_ARRAY_BUFFER, numf*mNumSides*sizeof(int), &mFaceVN[0], GL_STATIC_DRAW_ARB);
		glBindVertexArray ( 0 );

	#endif
}
void nvMesh::SelectVAO ()
{
	glBindVertexArray ( mVAO );
}

void nvMesh::SelectVBO ( )
{
	#ifdef USE_DX
		#ifdef DEBUG_UTIL
			nvprintf  ( "nvMesh: SelectVBO\n" );
		#endif
		UINT stride[3];	
		UINT offset[3];
		ID3D11Buffer* vptr[3];
		vptr[0] = mVBO[0];		stride[0] = sizeof(Vertex);		offset[0] = 0;		// Pos		
		vptr[1] = mVBO[0];		stride[1] = sizeof(Vertex);		offset[1] = 12;		// Normal
		vptr[2] = mVBO[0];		stride[2] = sizeof(Vertex);		offset[2] = 24;		// UV
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 				
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST ); 	
		g_pContext->IASetIndexBuffer ( mVBO[1], DXGI_FORMAT_R32_UINT, 0 );
		checkDEV( "nvMesh: SelectVBO" );
	#else
		glBindBuffer ( GL_ARRAY_BUFFER, mVBO[0] );		
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localClr, 4, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localUV,  2, GL_FLOAT, false, sizeof(Vertex), (void*) 24 );
		glVertexAttribPointer ( localNorm,3, GL_FLOAT, false, sizeof(Vertex), (void*) 12 );		
		glEnableVertexAttribArray ( localPos );	
		glEnableVertexAttribArray ( localUV );
		glEnableVertexAttribArray ( localNorm );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );			
	#endif
}
void nvMesh::Draw ( int inst )
{
	#ifdef USE_DX		
		#ifdef DEBUG_UTIL
			nvprintf  ( "nvMesh: DrawIndexedInstanced\n" );
		#endif
		g_pContext->DrawIndexedInstanced ( mNumFaces*mNumSides, inst, 0, 0, 0 );
		checkDEV( "nvMesh: DrawIndexedInstanced" );
	#else		
		glDrawElementsInstanced ( (mNumSides==3) ? GL_TRIANGLES : GL_QUADS, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
	checkGL ( "nvMesh:Draw" );
}
void nvMesh::DrawPatches ( int inst )
{
	#ifdef USE_DX
	#else
		glPatchParameteri( GL_PATCH_VERTICES, mNumSides );
		glDrawElementsInstanced ( GL_PATCHES, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
}

