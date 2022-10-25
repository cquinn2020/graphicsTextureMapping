#pragma once

#include <stdlib.h>
#include <math.h>
#include <vector>

#include <glm/glm.hpp>

class Triangle
{
private:
	glm::vec3 v[3]; // Triangle vertices
	glm::vec3 c[3]; // Vertex color
	glm::vec2 t[3]; // Texture coordinates

public:
	// Default constructor
	Triangle();

	// Constructor without texture coordinates
	Triangle(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, const int &color);

	// Constructor with texture coordinates
	Triangle(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2, glm::vec2 &t0, glm::vec2 &t1, glm::vec2 &t2, const int &color);

	// Rendering the triangle using OpenGL
	void RenderOpenGL(glm::mat4 &modelViewMatrix, glm::mat4 &projectionMatrix, bool textureMode);

	// Rendering the triangle using CPU and add color as parameter (float color[WINDOW_HEIGHT][WINDOW_WIDTH][3];)
	void RenderCPU(glm::mat4 &modelViewMatrix, glm::mat4 &projectionMatrix, float color[1024][1024][3], float depth[1024][1024], int textWidth, int textHeight, bool textureMode, std::vector<float *> texture, int textureNum);
};
