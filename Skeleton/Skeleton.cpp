//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Pelyhe Adam
// Neptun : u0x77g
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";


// fragment shader in GLSL

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	uniform int colorOrTexture;		// if not 0, use texturing	
	uniform vec3 color;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		if (colorOrTexture != 0) {
			fragmentColor = texture(textureUnit, texCoord);
		} else {
			fragmentColor = vec4(color, 1);			// computed 
		}
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vaoNode;	   // virtual world on the GPU
unsigned int vaoEdge;	   // virtual world on the GPU

vec2 oldMousePosition;
vec2 mousePosition;


class Graph {
public:

	vec2 nodes2D[50];		// 50 points with x and y coordinates
	vec3 nodes3D[50];
	int neighbourMatrix[50][50] = { { 0 } };
	vec2 edges[61 * 2];		// 5% fullness means 61 edges, and each one has 2 endpoints
	Texture* textures[50];

	Graph() {
		generateGraph();
		generateTextures();
	}

	void draw() {
		updateGraph();
		drawNodes();
		drawEdges();
	}

	void forceDirectedAlgorithm() {
		const float optimalDistance = 0.3f;
		
		for (int i = 0; i < 50; i++) {
			vec2 force(0, 0);
			for (int j = 0; j < 50; j++) {
				if (i == j) {
					continue;
				}
				float distance = distanceBetweenTwoPoints(nodes3D[i], nodes3D[j]);
				float p = distance / optimalDistance;

				if (neighbourMatrix[i][j] == 1 || neighbourMatrix[j][i] == 1) {
					force = ((force + forceBetweenNeighbours(nodes2D[j], nodes2D[i], p, distance)));
				}
				else if (neighbourMatrix[i][j] == 0 && neighbourMatrix[j][i] == 0) {
					force = ((force + forceBetweenNotNeighbours(nodes2D[j], nodes2D[i], p, distance)));
				}
			}
			force = force + (-nodes2D[i] *0.7 );	// the force which keep the graph centered
			force = force * 0.02f;
			shiftOneNode(i, force);
		}

		//updateEdges();
		glutPostRedisplay();
	}

	void shift() {
		vec3 O(0.0f, 0.0f, 1.0f);
		vec3 Q;
		Q.x = mousePosition.x / sqrtf(1.0f - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));
		Q.y = mousePosition.y / sqrtf(1.0f - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));
		Q.z = 1.0f / sqrtf(1.0f - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));

		//printf("%f Q MOUSE DELTA X  %f Q MOUSE DELTA Y   %f Q MOUSE DELTA Z\n", Q.x, Q.y, Q.z);
		float distOQ = acoshf(-lorentz(Q, O));

		if (distOQ == 0.0f) {		// to avoid dividing by zero
			return;
		}

		vec3 V = (Q - (O * coshf(distOQ))) / sinhf(distOQ);

		vec3 m1 = (O * coshf(distOQ / 4)) + (V * sinhf(distOQ / 4));

		vec3 m2 = (O * coshf((3 * distOQ) / 4)) + (V * sinhf((3 * distOQ) / 4));

		vec3 n;
		for (int i = 0; i < 50; i++) {

			n.x = nodes3D[i].x;
			n.y = nodes3D[i].y;
			n.z = nodes3D[i].z;

			//printf("FORCIKLUS        %i %f:x    %f:y    %f:z\n",i, n.x, n.y, n.z);

			float distnm1 = acoshf(-lorentz(m1, n));

			if (distnm1 == 0.0f) {		// to avoid dividing by zero
				return;
			}

			vec3 V1 = (m1 - (n * coshf(distnm1))) / sinhf(distnm1);
			vec3 n1 = (n * coshf(distnm1 * 2)) + (V1 * sinhf(distnm1 * 2));

			float distnm2 = acoshf(-lorentz(m2, n1));

			if (distnm2 == 0.0f) {		// to avoid dividing by zero
				return;
			}

			vec3 V2 = (m2 - (n1 * coshf(distnm2))) / sinhf(distnm2);
			vec3 n2 = (n1 * coshf(distnm2 * 2)) + (V2 * sinhf(distnm2 * 2));

			nodes3D[i].x = n2.x;
			nodes3D[i].y = n2.y;
			nodes3D[i].z = n2.z;
		}

	}

private:
	void generateGraph() {

		int remainingEdges = 61;			// first there will be 61 edges and after generating edges this will be decremented
		// generating neighbour matrix
		while (true) {
			
			int i = rand() % 50;			// a random point of the matrix
			int j = rand() % 50;			// another random point of the matrix

			if (i != j && neighbourMatrix[i][j] == 0 && neighbourMatrix[j][i] == 0) {
				//printf("%i i %i j\n", i, j);
				neighbourMatrix[i][j] = 1;
				remainingEdges--;
			}

			if (remainingEdges == 0) 
				break;
		}
	

		// these variables are temporary beacuse the loop will run 50 times 
		// and from these graphs, the the one which crossing number is the lowest
		// will be selected

		vec2 nodes2DTemp[50];		
		vec3 nodes3DTemp[50];
		vec2 edgesTemp[61 * 2];		

		// the variable which contains the crossing number 
		int intersectPoints = 0;

		// the best graph will be selected from 100 
		for (int tries = 0; tries < 100; tries++) {

			// generating graph with x, y, z coordinates
			for (int i = 0; i < 50; i++) {
				float randomX = ((float)rand() / RAND_MAX) * (1.000f - (-1.000f)) + (-1.000f);		//random x coordinate
				float randomY = ((float)rand() / RAND_MAX) * (1.000f - (-1.000f)) + (-1.000f);		//random y coordinate
				nodes3DTemp[i].x = randomX;
				nodes3DTemp[i].y = randomY;
				nodes3DTemp[i].z = sqrtf(nodes3DTemp[i].x * nodes3DTemp[i].x + nodes3DTemp[i].y * nodes3DTemp[i].y + 1.0f);
			}

			// making a 2D graph from the hiperbolic one
			for (int i = 0; i < 50; i++) {
				nodes2DTemp[i].x = nodes3DTemp[i].x / nodes3DTemp[i].z;
				nodes2DTemp[i].y = nodes3DTemp[i].y / nodes3DTemp[i].z;
				//printf("[%i.] %f x coordinate %f y coordinate\n", i, nodes2DTemp[i].x, nodes2DTemp[i].y);
			}

			int index = 0;
			//adding the edges to the edges array according to the neighbour matrix
			for (int i = 0; i < 50; i++) {				
				for (int j = 0; j < 50; j++) {
					if (neighbourMatrix[i][j] == 1 ) {
						//printf("%i", index);
						edgesTemp[index++] = nodes2DTemp[i];
						edgesTemp[index++] = nodes2DTemp[j];
					}
				}
			}

			int temp = 0;
			// checking every edges whether they have a common point or not
			// if we compare one edge with itself, it the intersectCheck() will return with false
			for (int j = 0; j < 122; j+=2) {
				for (int i = 0; i < 122; i+=2) {
					bool intersect = intersectCheck(edgesTemp[j], edgesTemp[j + 1], edgesTemp[i], edgesTemp[i + 1]);
					if (intersect) {
						temp++;
					}
				}
			}
			// if it's the first generated graph (beacuse the intersectPoints is initialized as 0)
			// or the generated graph's crossing number is less then the previous one, 
			// then change the old graph for the new one
			if (intersectPoints == 0 || temp < intersectPoints) {
				for (int i = 0; i < 50; i++) {
					nodes2D[i] = nodes2DTemp[i];
					nodes3D[i] = nodes3DTemp[i];
					//printf("[%i.] %f x coordinate %f y coordinate\n", i, nodes2D[i].x, nodes2D[i].y);
				}
				for (int i = 0; i < 121; i+=2) {
					edges[i] = edgesTemp[i];
					//printf("%i %f:x %f:y   ---    %f:x %f:y\n", i, edgesTemp[i].x, edges[i].y, edgesTemp[i+1].x, edgesTemp[i+1].y);
				}
				//printf("%i <--- régi %i <----- új \n", intersectPoints, temp);
				intersectPoints = temp;
			}
		}
		
	}

	void generateTextures() {
		int width = 8, height = 8;				// create checkerboard texture procedurally
		std::vector<vec4> image(width * height);

		for (int i = 0; i < 50; i++) {				
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					float red = ((float)rand() / RAND_MAX);	//random red component
					float green = ((float)rand() / RAND_MAX);		//random green component
					float blue = ((float)rand() / RAND_MAX);		//random blue component
					image[y * width + x] = vec4(red, green, blue, 1);			//creating an image, with the random color
				}
			}
			textures[i] = new Texture(width, height, image);		// add the image to the textures array
		}

	}
	
	void updateNodes() {
		for (int i = 0; i < 50; i++) {
			nodes2D[i].x = nodes3D[i].x / nodes3D[i].z;
			nodes2D[i].y = nodes3D[i].y / nodes3D[i].z;
		}
	}

	void updateEdges() {
		int index = 0;
		//adding the edges to the edges array according to the neighbour matrix
		for (int i = 0; i < 50; i++) {
			for (int j = 0; j < 50; j++) {
				if (neighbourMatrix[i][j] == 1) {
					//printf("%i", index);
					edges[index++] = nodes2D[i];
					edges[index++] = nodes2D[j];
				}
			}
		}
	}

	void updateGraph() {
		updateNodes();
		updateEdges();
	}

	void drawNodes() {
		// Set color to (0, 1, 0) = green
		changeColor(1.0f, 0.0f, 0.0f);

		vec2 circlePoints[100];
		
		vec2 vertices[4];
		vec2 uv[4];

		for (int j = 0; j < 50; j++) {
			for (int i = 0; i < 100; i++) {
				float fi = (float)(i * 2 * M_PI) / 100;
				circlePoints[i] = vec2(vec2(cosf(fi) * 0.05f, sinf(fi) * 0.05f) + vec2(nodes3D[j].x, nodes3D[j].y));
				float z = (float)sqrtf(1 + circlePoints[i].x * circlePoints[i].x + circlePoints[i].y * circlePoints[i].y);
				circlePoints[i] = circlePoints[i] / z;
			}

			vertices[0] = circlePoints[24];
			vertices[1] = circlePoints[49];
			vertices[2] = circlePoints[74];
			vertices[3] = circlePoints[99];

			gpuProgram.setUniform(0, "colorOrTexture");

			glBindVertexArray(vaoNode);  // Draw call
			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				sizeof(circlePoints),  // # bytes
				circlePoints,	      	// address
				GL_STATIC_DRAW);	// we do not change later

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed
			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 100 /*# Elements*/);

			gpuProgram.setUniform((*textures[j]), "textureUnit");
			gpuProgram.setUniform(1, "colorOrTexture");
			glBufferData(GL_ARRAY_BUFFER,
				sizeof(vertices),
				vertices,
				GL_STATIC_DRAW);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0,
				2, GL_FLOAT, GL_FALSE,
				0, NULL);
			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 4 /*# Elements*/);


		}

		gpuProgram.setUniform(0, "colorOrTexture");
	}

	void drawEdges() {
		// Set color to (1, 1, 0) = yellow
		changeColor(1.0f, 1.0f, 0.0f);

		glBindVertexArray(vaoEdge);  // Draw call
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(edges),  // # bytes
			edges,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
		glDrawArrays(GL_LINES, 0, 122);
	}

	void changeColor(float r, float g, float b) {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, r, g, b); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
	}

	//source: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
	bool intersectCheck(vec2 edge1Start, vec2 edge1End, vec2 edge2Start, vec2 edge2End) {

		//if we want to compare the same two edges, return false
		if (edge1Start.x == edge2Start.x && edge1Start.y == edge2Start.y
			&& edge1End.x == edge2End.x && edge1End.y == edge2End.y) {
			return false;
		}

		float maxXedge1 = maximum(edge1Start.x, edge1End.x);		float minXedge1 = minimum(edge1Start.x, edge1End.x);
		float maxXedge2 = maximum(edge2Start.x, edge2End.x);		float minXedge2 = minimum(edge2Start.x, edge2End.x);

		// two line can't intersect if they have no common points on the x axis.
		// the maximum of the minimums of the two points x coordinates has to be smaller
		// than the mimimum of the maxiumums of the two points!
		// otherwise there is no way for intersection
		if (maximum(minXedge1, minXedge2) > minimum(maxXedge1, maxXedge2)) {
			return false;
		}

		// the formulas are the following:
		// f1(x) = A1*x + b1 = y
		// f2(x) = A2*x + b2 = y
		float A1 = 0, A2 = 0;

		// gradient of edge1
		if ((edge1Start.x - edge1End.x) != 0) {
			A1 = (edge1Start.y - edge1End.y) / (edge1Start.x - edge1End.x);
		}

		// gradient of edge2
		if ((edge1Start.x - edge1End.x) != 0) {
			A2 = (edge1Start.y - edge1End.y) / (edge1Start.x - edge1End.x);
		}

		// one random point of the line is required (now it is the start point in both cases):
		float b1 = (edge1Start.y - A1 * edge1Start.x);
		float b2 = (edge2Start.y - A2 * edge2Start.x);

		// if they intersect, they intersect in P1 point (with xP, yP coordinates), which means:
		// yP = A1 * xP + b1
		// yP = A2 * xP + b2
		// --> A1 * xP + b1 = A2 * xP + b2
		// --> xP = (b2 - b1) / (A1 - A2)
		float xP = 0.0f;
		if (A1 - A2 != 0) {				// to avoid dividing by zero
			float xP = (b2 - b1) / (A1 - A2);
		}
		

		if ((xP < maximum(minXedge1, minXedge2)) || (xP > minimum(maxXedge1, maxXedge2))) {
			return false;
		}
		else {
			return true;
		}

	}

	// I can't add new library, thus I created a minimum 
	// and a maximum function, which are similar to std::min and std::max
	float maximum(float a, float b) {
		return (a < b) ? b : a;
	}

	float minimum(float a, float b) {
		return (a > b) ? b : a;
	}

	float lorentz(vec3 p1, vec3 p2) {
		return p1.x * p2.x + p1.y * p2.y - p1.z * p2.z;
	}

	float distanceBetweenTwoPoints(vec3 p1, vec3 p2) {
		return acosh(-lorentz(p1, p2));
	}

	vec2 forceBetweenNeighbours(vec2 p1, vec2 p2, float p, float distance) {
		return (p1 - p2) * log(p);
	}

	vec2 forceBetweenNotNeighbours(vec2 p1, vec2 p2, float p, float distance) {
		return (p1 - p2) * (-1 / (p * p));
	}

	void shiftOneNode(int index, vec2 force) {
		vec3 O(0.0f, 0.0f, 1.0f);
		vec3 Q;
		Q.x = force.x / sqrtf(1.0f - (force.x * force.x) - (force.y * force.y));
		Q.y = force.y / sqrtf(1.0f - (force.x * force.x) - (force.y * force.y));
		Q.z = 1.0f / sqrtf(1.0f - (force.x * force.x) - (force.y * force.y));

		//printf("%f Q MOUSE DELTA X  %f Q MOUSE DELTA Y   %f Q MOUSE DELTA Z\n", Q.x, Q.y, Q.z);
		float distOQ = acoshf(-lorentz(Q, O));

		if (distOQ == 0.0f) {		// to avoid dividing by zero
			return;
		}

		vec3 V = (Q - (O * coshf(distOQ))) / sinhf(distOQ);

		vec3 m1 = (O * coshf(distOQ / 4)) + (V * sinhf(distOQ / 4));

		vec3 m2 = (O * coshf((3 * distOQ) / 4)) + (V * sinhf((3 * distOQ) / 4));

		vec3 n;

			n.x = nodes3D[index].x;
			n.y = nodes3D[index].y;
			n.z = nodes3D[index].z;

			//printf("FORCIKLUS        %i %f:x    %f:y    %f:z\n",i, n.x, n.y, n.z);

			float distnm1 = acoshf(-lorentz(m1, n));

			if (distnm1 == 0.0f) {		// to avoid dividing by zero
				return;
			}

			vec3 V1 = (m1 - (n * coshf(distnm1))) / sinhf(distnm1);
			vec3 n1 = (n * coshf(distnm1 * 2)) + (V1 * sinhf(distnm1 * 2));

			float distnm2 = acoshf(-lorentz(m2, n1));

			if (distnm2 == 0.0f) {		// to avoid dividing by zero
				return;
			}

			vec3 V2 = (m2 - (n1 * coshf(distnm2))) / sinhf(distnm2);
			vec3 n2 = (n1 * coshf(distnm2 * 2)) + (V2 * sinhf(distnm2 * 2));

			nodes3D[index].x = n2.x;
			nodes3D[index].y = n2.y;
			nodes3D[index].z = n2.z;
		}

	
};

Graph* graph;
bool callFunction = false;
int iter = 0;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(7.0);		//source: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glPointSize.xml
	glLineWidth(2.0);

	glGenVertexArrays(1, &vaoNode);	// get 1 vao id
	glBindVertexArray(vaoNode);		// make it active

	unsigned int vboNode;		// vertex buffer object
	glGenBuffers(1, &vboNode);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vboNode);

	glGenVertexArrays(1, &vaoEdge);	// get 1 vao id
	glBindVertexArray(vaoEdge);		// make it active

	unsigned int vboEdge;		// vertex buffer object
	glGenBuffers(1, &vboEdge);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vboEdge);
	
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");

	graph = new Graph();

}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	graph->draw();
	glutSwapBuffers(); // exchange buffers for double buffering 
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ' ) {
		callFunction = true;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (cX * cX + cY * cY >= 1)
		return;

	//printf("%f x  %f y\n", oldMousePosition.x, oldMousePosition.y);
	mousePosition.x = cX - oldMousePosition.x;
	mousePosition.y = cY - oldMousePosition.y;

	oldMousePosition.x = cX;
	oldMousePosition.y = cY;
	
	//printf("\n%f : MOUSEDELTA.X     %f: MOUSEDELTA.Y\n", oldMousePosition.x, oldMousePosition.y);
	graph->shift();
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (cX * cX + cY * cY >= 1)
		return;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: 
		buttonStat = "pressed"; 
		oldMousePosition.x = cX;
		oldMousePosition.y = cY;
		break;
	case GLUT_UP:   
		buttonStat = "released";
		break;
	}

	glutPostRedisplay();
}


// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program

	if (callFunction) {
		iter++;
		graph->forceDirectedAlgorithm();
		if (iter == 1000) {
			callFunction = false;
			iter = 0;
		}
	}

}


// Key of ASCII code pressed
