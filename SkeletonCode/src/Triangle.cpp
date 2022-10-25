#include "Triangle.h"
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <iostream>
// A function clamping the input values to the lower and higher bounds
#define CLAMP(in, low, high) ((in) < (low) ? (low) : ((in) > (high) ? (high) : in))

glm::mat4 buildViewportMatrix(int width, int height)
{
	glm::mat4 viewportMatrix = glm::mat4(0.0f);
	viewportMatrix[0][0] = width / 2.0f;
	viewportMatrix[0][3] = width / 2.0f;
	viewportMatrix[1][1] = height / 2.0f;
	viewportMatrix[1][3] = height / 2.0f;
	viewportMatrix[2][2] = 1;
	viewportMatrix[3][3] = 1;
	viewportMatrix = glm::transpose(viewportMatrix);
	return viewportMatrix;
}
void computeAlphaBetaGamma(int i, int j, glm::vec3 v0prime, glm::vec3 v1prime, glm::vec3 v2prime, float &alpha, float &beta, float &gamma)
{
	alpha = (-((i + .5) - v1prime.x) * (v2prime.y - v1prime.y) + ((j + .5) - v1prime.y) * (v2prime.x - v1prime.x)) /
			(-(v0prime.x - v1prime.x) * (v2prime.y - v1prime.y) + (v0prime.y - v1prime.y) * (v2prime.x - v1prime.x));
	beta = (-((i + .5) - v2prime.x) * (v0prime.y - v2prime.y) + ((j + .5) - v2prime.y) * (v0prime.x - v2prime.x)) /
		   (-(v1prime.x - v2prime.x) * (v0prime.y - v2prime.y) + (v1prime.y - v2prime.y) * (v0prime.x - v2prime.x));
	gamma = 1 - alpha - beta;
}

glm::vec3 BilinearInterpolation(glm::vec2 &upvp, std::vector<float *> texture, int level, int textWidth)
{
	glm::vec2 u00 = glm::vec2(std::floor(upvp.x), std::floor(upvp.y));
	glm::vec2 u01 = glm::vec2(std::floor(upvp.x), std::ceil(upvp.y));
	glm::vec2 u10 = glm::vec2(std::ceil(upvp.x), std::floor(upvp.y));
	glm::vec2 u11 = glm::vec2(std::ceil(upvp.x), std::ceil(upvp.y));

	float s = upvp.x - u00.x;
	float t = upvp.y - u00.y;

	// get colors at the four corners
	float r00 = texture[level][(int)std::floor(u00.y * textWidth * 3 + u00.x * 3)];
	float g00 = texture[level][(int)std::floor(u00.y * textWidth * 3 + u00.x * 3 + 1)];
	float b00 = texture[level][(int)std::floor(u00.y * textWidth * 3 + u00.x * 3 + 2)];

	glm::vec3 c00 = glm::vec3(r00, g00, b00);

	float r01 = texture[level][(int)std::floor(u01.y * textWidth * 3 + u01.x * 3)];
	float g01 = texture[level][(int)std::floor(u01.y * textWidth * 3 + u01.x * 3 + 1)];
	float b01 = texture[level][(int)std::floor(u01.y * textWidth * 3 + u01.x * 3 + 2)];

	glm::vec3 c01 = glm::vec3(r01, g01, b01);

	float r10 = texture[level][(int)std::floor(u10.y * textWidth * 3 + u10.x * 3)];
	float g10 = texture[level][(int)std::floor(u10.y * textWidth * 3 + u10.x * 3 + 1)];
	float b10 = texture[level][(int)std::floor(u10.y * textWidth * 3 + u10.x * 3 + 2)];

	glm::vec3 c10 = glm::vec3(r10, g10, b10);

	float r11 = texture[level][(int)std::floor(u11.y * textWidth * 3 + u11.x * 3)];
	float g11 = texture[level][(int)std::floor(u11.y * textWidth * 3 + u11.x * 3 + 1)];
	float b11 = texture[level][(int)std::floor(u11.y * textWidth * 3 + u11.x * 3 + 2)];

	glm::vec3 c11 = glm::vec3(r11, g11, b11);

	// interpolate horizontally
	glm::vec3 u0 = c00 + s * (c10 - c01);
	glm::vec3 u1 = c01 + s * (c11 - c01);

	// interpolate vertically
	glm::vec3 c = u0 + t * (u1 - u0);

	return c;
}
// To implement mipmapping, you need to first find the right scale (L = log2 D). This can be done by
// the gradient formula in slide 66.
// • The calculated scale has a floating point value. You should first find the two closest levels to this.
// Then obtain the corresponding color for each level using bilinear interpolation. Finally, perform a
// linear interpolation between the two calculated colors to obtain the final color.
int computeMipLevel(glm::vec2 &upvp)
{
}

// texture coordinates could be less than 0 or greater than one. You need to map these values to the
// range[0, 1] using a wrap around function.
void mapTextureCoordinates(glm::vec2 &upvp)
{
	if (upvp.x < 0)
	{
		upvp = glm::vec2(1 + upvp.x, upvp.y);
	}
	if (upvp.x >= 1)
	{
		upvp = glm::vec2(upvp.x - 1, upvp.y);
	}
	if (upvp.y < 0)
	{
		upvp = glm::vec2(upvp.x, 1 + upvp.y);
	}
	if (upvp.y >= 1)
	{
		upvp = glm::vec2(upvp.x, upvp.y - 1);
	}
}

Triangle::Triangle()
{
	v[0] = glm::vec3(0.0f, 0.0f, 0.0f);
	v[1] = glm::vec3(0.0f, 0.0f, 0.0f);
	v[2] = glm::vec3(0.0f, 0.0f, 0.0f);

	c[0] = glm::vec3(0.0f, 0.0f, 0.0f);
	c[1] = glm::vec3(0.0f, 0.0f, 0.0f);
	c[2] = glm::vec3(0.0f, 0.0f, 0.0f);

	t[0] = glm::vec2(0.0f, 0.0f);
	t[1] = glm::vec2(0.0f, 0.0f);
	t[2] = glm::vec2(0.0f, 0.0f);
}

Triangle::Triangle(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, const int &color)
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;

	t[0] = glm::vec2(0.0f, 0.0f);
	t[1] = glm::vec2(0.0f, 0.0f);
	t[2] = glm::vec2(0.0f, 0.0f);

	// if the color mode is 0, then we want to make the triangle a random color
	if (color == 0)
	{
		double color = (float)rand() / (float)RAND_MAX;
		double color2 = (float)rand() / (float)RAND_MAX;
		double color3 = (float)rand() / (float)RAND_MAX;
		c[0] = glm::vec3(color, color2, color3);
		c[1] = glm::vec3(color, color2, color3);
		c[2] = glm::vec3(color, color2, color3);
	}
	// if the color mode is 1, then we want to make each vertex of the triangle a random color
	if (color == 1)
	{
		c[0] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		c[1] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		c[2] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
	}
	// if color is 2, then use each vertex's z value to determine the color
	if (color == 2)
	{ // https://math.stackexchange.com/questions/1567823/how-can-i-map-a-1-to-1-range-into-0-1
		float color1 = ((v0.z + 1) / 2);
		float color2 = ((v1.z + 1) / 2);
		float color3 = ((v2.z + 1) / 2);
		c[0] = glm::vec3(0, color1, 0);
		c[1] = glm::vec3(0, color2, 0);
		c[2] = glm::vec3(0, color3, 0);
	}
};

Triangle::Triangle(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec2 &t0, glm::vec2 &t1, glm::vec2 &t2, const int &color)
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;

	t[0] = t0;
	t[1] = t1;
	t[2] = t2;

	// if the color mode is 0, then we want to make the triangle a random color
	if (color == 0)
	{
		double color = (float)rand() / (float)RAND_MAX;
		double color2 = (float)rand() / (float)RAND_MAX;
		double color3 = (float)rand() / (float)RAND_MAX;
		c[0] = glm::vec3(color, color2, color3);
		c[1] = glm::vec3(color, color2, color3);
		c[2] = glm::vec3(color, color2, color3);
	}
	// if the color mode is 1, then we want to make each vertex of the triangle a random color
	if (color == 1)
	{
		c[0] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		c[1] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		c[2] = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
	}
	// if color is 2, then use each vertex's z value to determine the color
	if (color == 2)
	{ // https://math.stackexchange.com/questions/1567823/how-can-i-map-a-1-to-1-range-into-0-1
		float color1 = ((v0.z + 1) / 2);
		float color2 = ((v1.z + 1) / 2);
		float color3 = ((v2.z + 1) / 2);
		c[0] = glm::vec3(0, color1, 0);
		c[1] = glm::vec3(0, color2, 0);
		c[2] = glm::vec3(0, color3, 0);
	}
};

// Rendering the triangle using OpenGL
void Triangle::RenderOpenGL(glm::mat4 &modelViewMatrix, glm::mat4 &projectionMatrix, bool isTextured)
{

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(modelViewMatrix));

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(glm::value_ptr(projectionMatrix));

	// For textured object
	if (isTextured)
	{
		glEnable(GL_TEXTURE_2D);

		// Avoid modulating the texture by vertex color
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		glBegin(GL_TRIANGLES);

		glTexCoord2f(t[0].x, t[0].y);
		glVertex3f(v[0].x, v[0].y, v[0].z);

		glTexCoord2f(t[1].x, t[1].y);
		glVertex3f(v[1].x, v[1].y, v[1].z);

		glTexCoord2f(t[2].x, t[2].y);
		glVertex3f(v[2].x, v[2].y, v[2].z);

		glEnd();

		glDisable(GL_TEXTURE_2D);
	}
	// For object with only vertex color
	else
	{
		glBegin(GL_TRIANGLES);

		glColor3f(c[0].x, c[0].y, c[0].z);
		glVertex3f(v[0].x, v[0].y, v[0].z);

		glColor3f(c[1].x, c[1].y, c[1].z);
		glVertex3f(v[1].x, v[1].y, v[1].z);

		glColor3f(c[2].x, c[2].y, c[2].z);
		glVertex3f(v[2].x, v[2].y, v[2].z);

		glEnd();
	}
}

// Render the triangle on CPU (add color as parameter )
void Triangle::RenderCPU(glm::mat4 &modelViewMatrix, glm::mat4 &projectionMatrix, float color[1024][1024][3], float depth[1024][1024], int textWidth, int textHeight, bool textureMode, std::vector<float *> texture, int textureNum)
{
	// **** Transform the triangle to the screen **** /////
	int HEIGHT = 1024;
	int WIDTH = 1024;

	// multiply each vertex of triangle by the modelview and projection matrix (if not textured)
	auto v0 = projectionMatrix * modelViewMatrix * glm::vec4(v[0], 1.0f);
	auto v1 = projectionMatrix * modelViewMatrix * glm::vec4(v[1], 1.0f);
	auto v2 = projectionMatrix * modelViewMatrix * glm::vec4(v[2], 1.0f);

	// first do modelview * vertex and store each corresponding z value (for texture mapping)
	auto v0T = modelViewMatrix * glm::vec4(v[0], 1.0f);
	auto v1T = modelViewMatrix * glm::vec4(v[1], 1.0f);
	auto v2T = modelViewMatrix * glm::vec4(v[2], 1.0f);

	// store the z values in a vector (for texture mapping)
	std::vector<float> zValues;
	std::vector<float> zInverse;
	zValues.push_back(v0T.z);
	zValues.push_back(v1T.z);
	zValues.push_back(v2T.z);

	// store the inverse of the z values in a vector (for texture mapping)
	zInverse.push_back(1 / v0T.z);
	zInverse.push_back(1 / v1T.z);
	zInverse.push_back(1 / v2T.z);

	glm::vec2 IA = t[0] / zValues[0];
	glm::vec2 IB = t[1] / zValues[1];
	glm::vec2 IC = t[2] / zValues[2];

	// // divide each vertex x, y, z by its w value (for perspective correction)
	glm::vec3 v0prime = glm::vec3(v0.x / v0.w, v0.y / v0.w, v0.z / v0.w);
	glm::vec3 v1prime = glm::vec3(v1.x / v1.w, v1.y / v1.w, v1.z / v1.w);
	glm::vec3 v2prime = glm::vec3(v2.x / v2.w, v2.y / v2.w, v2.z / v2.w);

	glm::mat4 viewportMatrix = buildViewportMatrix(WIDTH, HEIGHT);

	v0prime = glm::vec3(viewportMatrix * glm::vec4(v0prime, 1.0f)); // A
	v1prime = glm::vec3(viewportMatrix * glm::vec4(v1prime, 1.0f)); // B
	v2prime = glm::vec3(viewportMatrix * glm::vec4(v2prime, 1.0f)); // C

	// **** Rasterize the triangle **** /////

	// make the bounding box
	int minX = (int)std::min(std::min(v0prime.x, v1prime.x), v2prime.x);
	int maxX = (int)std::max(std::max(v0prime.x, v1prime.x), v2prime.x) + 1;
	int minY = (int)std::min(std::min(v0prime.y, v1prime.y), v2prime.y);
	int maxY = (int)std::max(std::max(v0prime.y, v1prime.y), v2prime.y) + 1;

	float alpha;
	float beta;
	float gamma;
	float alphaAbove; // for texture mapping using mipmapping
	float betaAbove;
	float gammaAbove;
	float alphaRight; // for texture mapping using mipmapping
	float betaRight;
	float gammaRight;
	float depthValue;

	glm::vec2 Ip; // for textel at (i, j)
	glm::vec2 upvp;
	float zp;

	glm::vec2 IpAbove; // for textel at (i, j) above
	glm::vec2 upvpAbove;
	float zpAbove;

	glm::vec2 IpRight; // for textel at (i, j) right
	glm::vec2 upvpRight;
	float zpRight;

	for (int i = minX; i <= maxX; i++) // v0 (A), v1 (B), v2 (C)
	{
		for (int j = minY; j <= maxY; j++)
		{
			computeAlphaBetaGamma(i + .5, j + .5, v0prime, v1prime, v2prime, alpha, beta, gamma);				  // for textel at (i,j)
			computeAlphaBetaGamma(i + .5, j + 1.5, v0prime, v1prime, v2prime, alphaAbove, betaAbove, gammaAbove); // for textel above (i,j)
			computeAlphaBetaGamma(i + 1.5, j + .5, v0prime, v1prime, v2prime, alphaRight, betaRight, gammaRight); // for textel to the right of (i,j)

			depthValue = alpha * v0prime.z + beta * v1prime.z + gamma * v2prime.z;

			Ip = alpha * IA + beta * IB + gamma * IC; // for textel at (i,j)
			zp = alpha * (1 / zValues[0]) + beta * (1 / zValues[1]) + gamma * (1 / zValues[2]);
			upvp = Ip / zp;

			IpAbove = alphaAbove * IA + betaAbove * IB + gammaAbove * IC; // for textel above (i,j)
			zpAbove = alphaAbove * (1 / zValues[0]) + betaAbove * (1 / zValues[1]) + gammaAbove * (1 / zValues[2]);
			upvpAbove = IpAbove / zpAbove;

			IpRight = alphaRight * IA + betaRight * IB + gammaRight * IC; // for textel to the right of (i,j)
			zpRight = alphaRight * (1 / zValues[0]) + betaRight * (1 / zValues[1]) + gammaRight * (1 / zValues[2]);
			upvpRight = IpRight / zpRight;

			// compute texture coord at (x+1, y) and (x, y+1)
			// need to scale the texture coordinates to the size of the texture
			upvp = glm::vec2(upvp.x * (textWidth), upvp.y * (textHeight));
			upvpAbove = glm::vec2(upvpAbove.x * (textWidth), upvpAbove.y * (textHeight));
			upvpRight = glm::vec2(upvpRight.x * (textWidth), upvpRight.y * (textHeight));

			// texture coordinates could be less than 0 or greater than one. You need to map these values to the
			// range[0, 1] using a wrap around function.
			// mapTextureCoordinates(upvp);
			mapTextureCoordinates(upvpAbove);
			mapTextureCoordinates(upvpRight);

			float r = 0.0f;
			float g = 0.0f;
			float b = 0.0f;

			if (alpha > 0 && beta > 0 && gamma > 0 && alpha < 1 && beta < 1 && gamma < 1)
			{
				if (depthValue < depth[j][i])
				{
					if (textureMode)
					{
						if (textureNum == 0) // nearest neighbor
						{
							mapTextureCoordinates(upvp);
							upvp = glm::vec2(std::floor(upvp.x), std::floor(upvp.y));
							// compute r, g, b values
							r = texture[0][(int)std::floor((upvp.y * textWidth * 3 + upvp.x * 3 + 0))];
							g = texture[0][(int)std::floor((upvp.y * textWidth * 3 + upvp.x * 3 + 1))];
							b = texture[0][(int)std::floor((upvp.y * textWidth * 3 + upvp.x * 3 + 2))];
						}
						else if (textureNum == 1) // bilinear interpolation
						{						  // need 4 nearest neighbors to compute the color (u00, u01, u10, u11)
							mapTextureCoordinates(upvp);

							glm::vec3 colors = BilinearInterpolation(upvp, texture, 0, textWidth);
							r = colors.x;
							g = colors.y;
							b = colors.z;
						}
						else // mipmap - do bilinear interpolation on above and right
						{
							float lx = std::sqrt(std::pow((upvpRight.x - upvp.x), 2) + std::pow((upvpRight.y - upvp.y), 2));
							float ly = std::sqrt(std::pow((upvpAbove.x - upvp.x), 2) + std::pow((upvpAbove.y - upvp.y), 2));

							float l = std::max(lx, ly);
							float d = std::log2(l);

							// bound d to [0, 10]
							if (d < 0)
								d = 0;
							else if (d > 10)
								d = 10;

							int Db = std::floor(d);
							int Da = std::ceil(d);

							mapTextureCoordinates(upvp);

							glm::vec3 color1 = BilinearInterpolation(upvp, texture, Db, textWidth);
							glm::vec3 color2 = BilinearInterpolation(upvp, texture, Da, textWidth);

							glm::vec3 color = (d - Db) * color1 + (Da - d) * color2;

							r = color.x;
							g = color.y;
							b = color.z;
						}
					}
					else
					{
						// interpolate the color
						r = alpha * c[0].x + beta * c[1].x + gamma * c[2].x;
						g = alpha * c[0].y + beta * c[1].y + gamma * c[2].y;
						b = alpha * c[0].z + beta * c[1].z + gamma * c[2].z;
					}
					// set the color
					color[j][i][0] = r;
					color[j][i][1] = g;
					color[j][i][2] = b;

					// set the depth
					depth[j][i] = depthValue;
				}
			}
		}
	}
}