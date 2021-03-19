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
#include "framework.h";

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vaoNode;	   // virtual world on the GPU
unsigned int vaoEdge;	   // virtual world on the GPU

vec2 nodes2D[50];		// 50 points with x and y coordinates
vec3 nodes3D[50];		
const int requiredEdges = (50 * 49 / 2) * 0.05;		// number of edges required
int neighbourMatrix[50][50] = { 0 };
vec2 edges[requiredEdges * 2];		// one edge has 2 nodes

vec2 oldMousePosition;
vec2 mousePosition;

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

void generateGraph() {
	
	// generating graph with x, y, z coordinates
	for (int i = 0; i < 50; i++) {
		float randomX = ((float)rand() / RAND_MAX) * (1.000f - (-1.000f)) + (-1.000f);		//random x coordinate
		float randomY = ((float)rand() / RAND_MAX) * (1.000f - (-1.000f)) + (-1.000f);		//random y coordinate
		nodes3D[i].x = randomX;
		nodes3D[i].y = randomY;
		nodes3D[i].z = sqrt(nodes3D[i].x * nodes3D[i].x + nodes3D[i].y * nodes3D[i].y + 1);
	}

	// making a 2D graph from the hiperbolic one
	for (int i = 0; i < 50; i++) {
		nodes2D[i].x = nodes3D[i].x / nodes3D[i].z;
		nodes2D[i].y = nodes3D[i].y / nodes3D[i].z;
	}


	int remainingEdges = requiredEdges;
	while (true) {
		int i = rand() % 51;			// a random point of the matrix
		int j = rand() % 51;			// another random point of the matrix

		if (i != j && neighbourMatrix[i][j] == 0) {
			neighbourMatrix[i][j] = 1;
			remainingEdges--;
			if (remainingEdges == 0)
				break;
		}
	}

	int index = 0;
	for (int i = 0; i < 50; i++) {				//adding the edges to the edges array according to the neighbour matrix
		for (int j = 0; j < 50; j++) {
			if (neighbourMatrix[i][j] == 1) {
				edges[index++] = nodes2D[i];
				edges[index++] = nodes2D[j];
			}
		}
	}
}

float lorentz(vec3 p1, vec3 p2) {
	return p1.x * p2.x + p1.y * p2.y - p1.z * p2.z;
}


void shift() {
	vec3 O(0.0f, 0.0f, 1.0f);
	vec3 Q;
	Q.x = mousePosition.x / sqrt(1 - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));
	Q.y = mousePosition.y / sqrt(1 - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));
	Q.z = 1 / sqrt(1 - (mousePosition.x * mousePosition.x) - (mousePosition.y * mousePosition.y));

	//printf("%f Q MOUSE DELTA X  %f Q MOUSE DELTA Y   %f Q MOUSE DELTA Z\n", Q.x, Q.y, Q.z);
	float distOQ = acosh(-lorentz(Q, O));
	
	vec3 V;
	if (distOQ != 0) {
		V = (Q - (O * cosh(distOQ))) / sinh(distOQ);
	}

	vec3 m1 = (O * cosh(distOQ / 4)) + (V * sinh(distOQ / 4));

	vec3 m2 = (O * cosh((3 * distOQ) / 4)) + (V * sinh((3 * distOQ) / 4));

	vec3 n;
	for (int i = 0; i < 50; i++) {

		n.x = nodes3D[i].x;
		n.y = nodes3D[i].y;
		n.z = nodes3D[i].z;
		//printf("FORCIKLUS        %i %f:x    %f:y    %f:z\n",i, n.x, n.y, n.z);
		float distnm1 = acosh(-lorentz(m1, n));
		vec3 V1 = (m1 - (n * cosh(distnm1))) / sinh(distnm1);
		vec3 n1 = (n * cosh(distnm1 * 2)) + (V1 * sinh(distnm1 * 2));

		float distnm2 = acosh(-lorentz(m2, n1));
		vec3 V2 = (m2 - (n1 * cosh(distnm2))) / sinh(distnm2);
		vec3 n2 = (n1 * cosh(distnm2 * 2)) + (V2 * sinh(distnm2 * 2));

		nodes3D[i].x = n2.x;
		nodes3D[i].y = n2.y;
		nodes3D[i].z = n2.z;
	}

}



void drawGraph() {

	for (int i = 0; i < 50; i++) {
		nodes2D[i].x = nodes3D[i].x / nodes3D[i].z;
		nodes2D[i].y = nodes3D[i].y / nodes3D[i].z;
	}

	int index = 0;
	for (int i = 0; i < 50; i++) {				//adding the edges to the edges array according to the neighbour matrix
		for (int j = 0; j < 50; j++) {
			if (neighbourMatrix[i][j] == 1) {
				edges[index++] = nodes2D[i];
				edges[index++] = nodes2D[j];
			}
		}
	}

	// Set color to (0, 1, 0) = green
	changeColor(1.0f, 0.0f, 0.0f);

	glBindVertexArray(vaoNode);  // Draw call
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(nodes2D),  // # bytes
		nodes2D,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed
	glDrawArrays(GL_POINTS, 0 /*startIdx*/, 50 /*# Elements*/);
	
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
	glDrawArrays(GL_LINES, 0, requiredEdges);

}




//							^^ helper functions ^^ 
// <------------------------------------------------------------------------>

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(7.0);		//source: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glPointSize.xml
	glLineWidth(0.25);

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
	
	generateGraph();
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}



// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	drawGraph();
	glutSwapBuffers(); // exchange buffers for double buffering 
}


void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}



// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;


	if (mousePosition.x * mousePosition.x + mousePosition.y * mousePosition.y >= 1)
		return;

	printf("%f x  %f y\n", oldMousePosition.x, oldMousePosition.y);
	mousePosition.x = cX - oldMousePosition.x;
	mousePosition.y = cY - oldMousePosition.y;

	oldMousePosition.x = cX;
	oldMousePosition.y = cY;
	
	//printf("\n%f : MOUSEDELTA.X     %f: MOUSEDELTA.Y\n", mouseDelta.x, mouseDelta.y);
	shift();


	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

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
}


// Key of ASCII code pressed
