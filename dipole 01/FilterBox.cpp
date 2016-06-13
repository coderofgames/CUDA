//--------------------------------------------------------------------------------------
//
// simple_glow sample
//
// Author: Tristan Lorach
// Email: tlorach@nvidia.com
//
// simple Glow effect, to show how to add a postprocessing effect over
// an existing sample without modifying it too deeply
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


#include <stdlib.h>
#include <stdio.h>
#include <map>

#include <GL/glew.h>
#include <GL/glut.h>

#include <nvSDKPath.h>

#include <Cg/CgGL.h>

#include "FilterBox.h"

static nv::SDKPath sdkPath;

FilterBox::FilterBox() : 
  bValid(false),
  vpx(0), vpy(0), vpw(0), vph(0),
  posx(0), posy(0),
  cgContext(NULL),
  cgEffect(NULL),
  cgTechnique(NULL),
  cgPassFilterH(NULL),
  cgPassFilterV(NULL),
  cgPassBlend(NULL),
  srcSampler(NULL),
  tempSampler(NULL),
  finalSampler(NULL)
{
}
FilterBox::~FilterBox()
{
}
void FilterBox::Destroy()
{
//    glDeleteRenderbuffersEXT(1, &depth_rb);
  glDeleteTextures(2, textureID);
//    glDeleteFramebuffersEXT(2, fb);
  cgDestroyContext(cgContext); 
}
/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
void CheckFramebufferStatus()
{
    GLenum status;
    status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            printf("Unsupported framebuffer format\n");
            fprintf(stderr, "Unsupported framebuffer format");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            printf("Framebuffer incomplete, missing attachment\n");
            fprintf(stderr, "Framebuffer incomplete, missing attachment");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            printf("Framebuffer incomplete, attached images must have same dimensions\n");
            fprintf(stderr, "Framebuffer incomplete, attached images must have same dimensions");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            printf("Framebuffer incomplete, attached images must have same format\n");
            fprintf(stderr, "Framebuffer incomplete, attached images must have same format");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            printf("Framebuffer incomplete, missing draw buffer\n");
            fprintf(stderr, "Framebuffer incomplete, missing draw buffer");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            printf("Framebuffer incomplete, missing read buffer\n");
            fprintf(stderr, "Framebuffer incomplete, missing read buffer");
            break;
        default:
            printf("Error %x\n", status);
			break;
    }
}
void FilterBox::initRT(int n, int w, int h)
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[n]);
  //
  // init texture
  //
  glBindTexture(GL_TEXTURE_2D, textureID[n]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
  //
  // SEEMS VERY IMPORTANT for the FBO to be valid. ARGL.
  //
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, textureID[n], 0);
  CheckFramebufferStatus();
}
/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
bool FilterBox::Initialize(int w, int h)
{
  bufw = w;
  bufh = h;
  //
  // FBO
  //
  glGenFramebuffersEXT(2, fb);
  glGenTextures(2, textureID);
  glGenRenderbuffersEXT(1, &depth_rb);
  initRT(0, w, h);
  initRT(1, w, h);
  //
  // initialize depth renderbuffer
  //
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_rb);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, w, h);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
                                GL_RENDERBUFFER_EXT, depth_rb);
  CheckFramebufferStatus();

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  //
  // CGFX things
  //
  cgContext = cgCreateContext();
  cgGLRegisterStates(cgContext);

  std::string resolved_path;

  if ( sdkPath.getFilePath( "src/simple_glow/FilterBox.cgfx", resolved_path)) {
        cgEffect = cgCreateEffectFromFile(cgContext, resolved_path.c_str(), NULL);
        if(!cgEffect)
        {
            const char * pszErrors = NULL;
            fprintf(stderr, "CgFx Parse error : %s", pszErrors);
            const char *listing = cgGetLastListing(cgContext);
            bValid = false;
            return false;
        }
  }
  else {
      fprintf( stderr, "Failed to find shader file '%s'\n", "src/simple_glow/FilterBox.cgfx");
      bValid = false;
      return false;
  }

  cgTechnique = cgGetNamedTechnique(cgEffect, "Filter");
  cgPassFilterH = cgGetNamedPass(cgTechnique, "verticalPass");
  cgPassFilterV = cgGetNamedPass(cgTechnique, "horizontalPass");
  cgPassBlend   = cgGetNamedPass(cgTechnique, "drawFinal");
  cgBlendFactor = cgGetNamedEffectParameter(cgEffect, "blendFactor");
  cgGlowFactor  = cgGetNamedEffectParameter(cgEffect, "glowFactor");
  srcSampler    = cgGetNamedEffectParameter(cgEffect, "srcSampler");
  tempSampler   = cgGetNamedEffectParameter(cgEffect, "tempSampler");
  finalSampler  = cgGetNamedEffectParameter(cgEffect, "finalSampler");
  verticalDir   = cgGetNamedEffectParameter(cgEffect, "verticalDir");
  horizontalDir = cgGetNamedEffectParameter(cgEffect, "horizontalDir");

  cgGLSetParameter2f(verticalDir, 0,1.0f/(float)h);
  cgGLSetParameter2f(horizontalDir, 1.0f/(float)w, 0);

  bValid = true;
  return true;
}

#define FULLSCRQUAD()\
  glBegin(GL_QUADS);\
  glTexCoord2f(0,0);\
  glVertex4f(-1, -1, 0,1);\
  glTexCoord2f(1,0);\
  glVertex4f(1, -1,0,1);\
  glTexCoord2f(1,1);\
  glVertex4f(1, 1,0,1);\
  glTexCoord2f(0,1);\
  glVertex4f(-1, 1,0,1);\
  glEnd();

/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
void FilterBox::Draw(float f)
{
  if(!bValid)
    return;
  glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
  CGbool bRes;
  bRes = cgValidateTechnique(cgTechnique);
  if(!bRes)
  {
    bValid = false;
    const char * pszErrors = NULL;
    fprintf(stderr, "Validation of FilterRect failed");
    fprintf(stderr, "CgFx Parse error : %s", pszErrors);
    const char *listing = cgGetLastListing(cgContext);
    return;
  }
  //
  // intermediate stage : bluring horizontal
  //
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[1]);
  glPushAttrib(GL_VIEWPORT_BIT); 
  glViewport(0,0,bufw,bufh);

  cgGLSetupSampler(srcSampler, textureID[0]);
  cgSetPassState(cgPassFilterH);

  FULLSCRQUAD();

  glPopAttrib();
  //
  // intermediate stage : bluring vertical
  //
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[0]);
  glPushAttrib(GL_VIEWPORT_BIT); 
  glViewport(0,0,bufw,bufh);

  cgGLSetupSampler(srcSampler, textureID[1]);
  cgSetPassState(cgPassFilterV);

  FULLSCRQUAD();

  glPopAttrib();
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  //
  // Final stage : Blend the final texture to the screen
  //
  cgGLSetupSampler(tempSampler, textureID[0]);

  cgSetPassState(cgPassBlend);
  glBlendColor(f,f,f,f);

  float xoffset = -1.0f + 2.0f*(float)posx/(float)vpw;
  float yoffset = -1.0f + 2.0f*(float)posy/(float)vph;
  float xoffset2 = xoffset + 2.0f*(float)width/(float)vpw;
  float yoffset2 = yoffset + 2.0f*(float)height/(float)vph;

  glBegin(GL_QUADS);
  glTexCoord2f(0,0);
  glVertex4f(xoffset, yoffset, 0,1);
  glTexCoord2f(1,0);
  glVertex4f(xoffset2, yoffset,0,1);
  glTexCoord2f(1,1);
  glVertex4f(xoffset2, yoffset2,0,1);
  glTexCoord2f(0,1);
  glVertex4f(xoffset, yoffset2,0,1);
  glEnd();
  cgResetPassState(cgPassBlend);
  glBlendColor(0,0,0,0);
}
/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
void FilterBox::Activate(int x, int y, int w, int h)
{
  if(!bValid)
    return;
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[0]);

  glPushAttrib(GL_VIEWPORT_BIT); 
  posx = x;
  posy = y;
  if(w>0) width = w;
  if(h>0) height = h;
  float scw = ((float)bufw/(float)width);
  float sch = ((float)bufh/(float)height);
  glViewport((int)(-(float)x*scw), (int)(-(float)y*sch), (int)((float)vpw*scw), (int)((float)vph*sch));
}
/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
void FilterBox::Deactivate()
{
  if(!bValid)
    return;
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  glPopAttrib();
}
/*-------------------------------------------------------------------------

  -------------------------------------------------------------------------*/
void FilterBox::setWindowViewPort(int x, int y, int w, int h)
{
  vpx = x;
  vpy = y;
  vpw = w;
  vph = h;
}
