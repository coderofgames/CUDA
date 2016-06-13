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
#include <Cg/CgGL.h>
class FilterBox
{
public:
  FilterBox(); 
  ~FilterBox();

  bool Initialize(int w, int h);
  void setWindowViewPort(int x, int y, int w, int h);
  void Destroy();

  void Activate(int x, int y, int w=-1, int h=-1);
  void Deactivate();
  void Draw(float f);

  CGparameter   cgBlendFactor;
  CGparameter   cgGlowFactor;
protected:
  bool          bValid;

  int           vpx, vpy, vpw, vph;
  int           posx, posy, width, height;
  int           bufw, bufh;

  CGcontext     cgContext;
  CGeffect      cgEffect;
  CGtechnique   cgTechnique;
  CGpass        cgPassFilterH;
  CGpass        cgPassFilterV;
  CGpass        cgPassBlend;
  GLuint        textureID[2];
  CGparameter   srcSampler;
  CGparameter   tempSampler;
  CGparameter   finalSampler;
  CGparameter   verticalDir;
  CGparameter   horizontalDir;
  GLuint        fb[2];
  GLuint        depth_rb;

  void          initRT(int n, int w, int h);
};