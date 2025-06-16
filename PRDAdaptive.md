# Software Conclusion Report: Adaptive AR Window System

## **Software Structure**

The system is developed in Python, leveraging predefined computer vision models to promote rapid prototyping and modular architecture. All modules follow a unified structure, providing clear and consistent input-output interfaces for ease of integration and testing.

The system architecture includes the following components:

1. **Data Acquisition**
	- Interfaces with Hololens2 using dedicated libraries to stream RGBD video and auxiliary sensor data to a central server.
	- Ensures frame synchronization with precise timestamp metadata.
2. **Sensing Module**
	- Utilizes pretrained models for semantic segmentation, object detection, and depth estimation (e.g., PyTorch, ONNX Runtime).
	- Produces standardized outputs including visual features and semantic detections.
3. **Field Map Construction**
	- Generates a multi-layered, grid-based field map with discrete resolution optimized for edge computation.
	- Combines individual layers into a monolithic map via weighted synthesis.
4. **Field Processing**
	- Converts the field map into a scalar potential field.
	- Derives gradients to simulate physical force fields acting on the AR window.
5. **Motion Control**
	- Implements an overdamped spring model to achieve smooth, non-oscillatory window transitions.
	- Encapsulates logic within a state machine for robustness.
6. **UI Control Interface**
	- Facilitates server-to-device communication, transmitting calculated window positions and interaction signals to the Hololens2 client.

This modular design allows independent development and replacement of subcomponents, improving maintainability and scalability.

## **Core Software Modules and Functions**

### 1. **Sensing Module**

- **Function**
	 Detect and extract environmental features from visual input for downstream processing.
- **Explanation**
	 This module processes RGBD video captured from the Hololens2's egocentric viewpoint. It applies pretrained models to obtain critical visual features and semantic context needed for field map generation.
- **Input**
	- RGBD video stream (Hololens2)
- **Output**
	- Depth map (composited)
	- 2D bounding boxes
	- Tracked object instances
	- Semantic labels of obstacles
	- Saliency map
	- Optical flow

### 2. **Field Map Construction Module**

- **Function**
	 Convert sensory data into a structured, multi-layered cost map representing spatial and social constraints.
- **Explanation**
	 This module builds a layered field map from the sensing output. Each layer encodes distinct environmental semantics or behavioral rules. The individual layers are aggregated into a final monolithic field map. A defined `Layer` class governs layer structure, and synthesis logic manages weighted integration.
- **Input**
	- Visual features from the Sensing Module
- **Output**
	- Monolithic Field Map containing:
		- Static Map Layer
		- Obstacle Map Layer
		- Inflation Layer
		- Proxemic Layer
		- Rule Layers
		- Optional: Attraction/Repulsion Layers

### 3. **Scalar Potential Field Computation Module**

- **Function**
	 Derive a scalar potential field and its gradient vector field from the monolithic cost map.
- **Explanation**
	 This component models the monolithic map as a potential energy field. Gradients extracted from this field serve as force vectors guiding the motion of the AR window. The method mirrors principles of electromagnetism, treating the AR interface as a charged particle.
- **Input**
	- Monolithic Field Map
- **Output**
	- Scalar potential field
	- Gradient vector field

### 4. **Motion Control Module**

- **Function**
	 Translate environmental force vectors into physically consistent AR window motion.
- **Explanation**
	 Uses an overdamped spring model to achieve stable, smooth motion transitions. The system avoids oscillations and ensures consistent responsiveness. A state machine architecture supports real-time adjustments.
- **Input**
	- Current window position
	- Gradient vector field
- **Output**
	- Updated AR window position

## **Update Cycle**

- **Function**
	 Regulate update frequency and coordinate asynchronous data streams.
- **Explanation**
	 Modules are updated at a fixed 5 Hz cycle. As input streams (e.g., RGB at 30 Hz) operate at higher frequencies, a sampling strategy is employed to select appropriate frames and unify timing across all modules.
- **Input**
	- Asynchronous sensor streams
- **Output**
	- Time-synchronized data samples at 5 Hz

## **Key Design Characteristics**

- **Physics-Inspired Control**: Employs force-based dynamics to produce intuitive, natural window behavior.
- **Layered Architecture**: Supports extensibility and separation of concerns through modular map layers.
- **Real-Time Operation**: Ensures low-latency processing for interactive user experiences.
- **Contextual Awareness**: Incorporates semantic, spatial, and social cues into movement planning logic.
- **Streaming Manner**: Adopts streaming process with camera data input and calculation output.