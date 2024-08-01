## Notes for further tuning
### Parameters to tune (stanley_controller_pkg/config/parameters.yaml)
Current values of parameters correspond to parameter values the controller can run on with highest successful speed profiler settings    
| Parameter | Current value  | Comment |
|-------------------|-----------------|----------|
| Throttle gain     | 0.2  | Gain of proportional controller |
| Cross track gain  | 2.2  | Might be modified to take velocity into consideration |
| Heading error gain| 1.0  | Is probably not needed (why it is set to 1.0). But could be experimented with | 
| Heading look ahead| 27   | Look ahead for heading error because car needs to start turning before inside of curve |

### Speed profiler settings (speed_profiler/ros_interface.py)
Current values correspond to "highest speed" the current controller parameters are tuned for. 
| Parameter | Current value  | Target value |
|-------------------|-----------------|------------|
| speed limit       | 18              | 20         |
| lateral acc. limit| 8               | 12         |
| acc. min          | -6              | -13        |
| acc. max          | 6               | 15         | 

### Comments
General comments:
Further tuning seems to be limited by longitudinal control. In particular, when increasing acc. min or acc.max, the car does not deccelerate for curves in simulation. Try for example implementing integral part of controller. 

## DV-TEMPLATE-PACKAGE

Example of generic package for usage with DV pipeline.

## Table of Contents

- [Guidelines](#guidelines)
  - [Overview](#overview)
  - [Enforced](#enforced)
  - [Encouraged](#encouraged)
- [Development](#development)
  - [Creating a package](#creating-a-package)
  - [Graphical User Interface (GUI) Support](#graphical-user-interface-gui-support)
- [Deployment](#deployment)
  - [Prerequisites](#prerequisites)
  - [Building and Running](#building-and-running)
    - [Stopping and Cleanup](#stopping-and-cleanup)
- [Validation and Testing](#validation-and-testing)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Guidelines

### Overview

- The final pipeline will be constructed using Docker-compose, and will be based on the Docker-compose files from each package repo.
- The ROS master will run in a separate service (container) in the Docker-compose, but will be connected trough a Docker network.
- All packages will firstly be built trough a entrypoint script and then launched with a launchfile using the command instruction in the Docker-compose file.

### Enforced

- Repository name ending with "_pkg".

- Main branch always in a deployable state.

- [Dockerfile](Dockerfile) that installs all package dependencies and sets up the workspace directory.

- Working ROS package that can be built with ```catkin build``` using [docker-entrypoint.sh](docker-entrypoint.sh), functionality from this script is used in GitHub Actions workflows.

- [Docker-compose file](docker-compose.yml) that builds and launches the package.

### Encouraged

- Avoid hard-coding variables and instead utilize configuration files.

- Incorporating unit testing with github actions.

- [Visualize codebase](https://githubnext.com/projects/repo-visualization/) by adding `![Visualization of the codebase](./diagram.svg)` in the README.

## Development

This section provides step-by-step instructions on how to create a new ROS package and integrate it with Docker using this template.

### Creating a package

1. Choose a suitable Docker base image based on your package's requirements. In many cases, the 'ros-base' image is sufficient. It's preferable to use [ROS Noetic](https://hub.docker.com/layers/library/ros/noetic-ros-base-focal/images/sha256-650483554542d585af1f7c6fd09a4fb64d1492243f19010662e5434e1489a9d6?context=explore), but [Melodic](https://hub.docker.com/layers/library/ros/melodic-ros-base-bionic/images/sha256-96705e5651a424f55c89221f8dc76f79b55250584b6f953a220c8b1c31ba78bd?context=explore) is also an option. For packages related to machine learning, consider the [dustynv jetpack containers](https://github.com/dusty-nv/jetson-containers). Ensure that all the package dependencies are installed and set up in the [Dockerfile](Dockerfile). Also, remember to adjust the ```ROS_DISTRO``` variables in accordance with the ROS distribution you're using. Refer to the provided [Dockerfile](Dockerfile) for guidance.
2. Ensure that the ROS package is complete and can be built using ```catkin build```. This entails adding source code, primarily in C++, to the ```src``` directory and incorporating Python scripts in the ```scripts``` directory. Additionally, modifications should be made to [package.xml](package.xml), [CMakeList.txt](CMakeLists.txt), [docker-entrypoint.sh](docker-entrypoint.sh), and [launch.launch](launch/launch.launch) (rename this launchfile to a suitable name of your package). Refer to the provided files and the [ROS wiki](https://wiki.ros.org/Documentation) for guidance.
3. Update the Docker-compose file. Ensure that the service name aligns with the package's name and modify the launch command to the correct package and launch file name. Also, set the `ROS_DISTRO` variable appropriately, this is used in the CI workflow. Additionally, if required, include extra mounted volumes in the `src` directory or adjust other arguments as appropriate.
4. Test the package compose file using `docker-compose up`. Make sure that the package in launched correctly, for example with `docker exec ros_container source devel/setup.bash && rosservice list`. See Usage for further guidance. This testing will soon be encorperated in a CI workflow.

### Graphical User Interface (GUI) Support

The `docker-compose.yml` includes a `noVNC` service that enables GUI support through X11 forwarding. The service is based on the [theasp/novnc](https://hub.docker.com/r/theasp/novnc) image. The service allows graphical applications to be run in your web browser by connecting to the container's VNC server (default `http://localhost:8080/vnc.html`).

For example, you might want to run `rviz` in your ROS container, this solution allows you to do so. To do this, you need to:

1. Enter the container:

   ```bash
   docker exec -it <container id or name> /bin/bash
   ```

2. Run `rviz` (or any other GUI application):

    ```bash
    rviz
    ```

3. Open a web browser and navigate to `http://localhost:8080/vnc.html`.

You can now interact with the GUI application in your browser for development or for other purposes.

## Deployment

This section provides step-by-step instructions on how to test and deploy the `dv-template-package` using Docker.

### Prerequisites

1. Ensure that the Dockerfile, docker-compose, launch files configuration and environment variables are set up as per the [package requirements](#creating-a-package).

2. Make sure you have Docker, or another container runtime, along with Docker-compose, installed on your machine. If not, you can get them from [here](https://docs.docker.com/compose/install/).

### Building and Running

1. **Clone the Repository**:

    ```bash
    git clone <your-package-repo-url>
    cd <your-package-repo-name>
    ```

2. **Build the Docker Image**:

   ```bash
   docker-compose build
   ```

3. **Run the Package**:

   ```bash
   docker-compose up
   ```

   This will execute the package as described in the `docker-compose.yml` file. Your ROS package should now be up and running within the Docker container. If this command is run before `docker-compose build`, it will automatically execute that as well.

4. To interact or debug within the running container:

   ```bash
   docker exec -it <container id or name> /bin/bash
   ```

   To get the name and id of all containers, run:

   ```bash
   docker ps
   ```

   To list all containers, including those that have exited, run:

   ```bash
    docker ps -a
    ```

#### Stopping and Cleanup

1. To stop the running services:

   ```bash
   docker-compose down
   ```

2. If you wish to remove the built images to free up space:

   ```bash
   docker rmi <image-name>
   ```

## Validation and Testing

### Testing manually

After launching the package, it's essential to validate its functionality:

1. **Check Package Health**:

   ```bash
   docker exec container_name bash -c "source devel/setup.bash && rosservice list"
   ```

   This should list the services exposed by your ROS nodes, indicating they're running correctly.

2. If unit tests have been integrated, run them to ensure package integrity:

   ```bash
   docker exec container_name <your-test-command>
   ```

### GitHub CI Workflow

To ensure that the package is functional and can be built successfully, a GitHub CI workflow is included in the repository. This workflow is triggered whenever a pull request is opened, updated, or reopened. The workflow is defined in the [`.github/workflows/ci.yml`](.github/workflows/ci.yml) file. Below is a brief description of the workflow.

**Trigger:**

- The CI workflow gets initiated whenever a pull request (PR) is:
  - Opened
  - Updated (synchronized)
  - Reopened
  
**Jobs:**

1. **Check for Essential Files**:
    - Ensure the presence of `Dockerfile`, `docker-compose.yml`, and `docker-entrypoint.sh` in the repository.

2. **Docker Operations**:
    - After verifying essential files, execute Docker Compose.

3. **Container Health Check**:
    - Wait and check if the Docker container is running, with a maximum wait time defined by `COMPOSE_UP_TIMEOUT_SECONDS`.

4. **Build Validation**:
    - After a short initial wait (5 seconds), inspect the Docker container for a sentinel file named `CONTAINER_INITIALIZED_PLACEHOLDER` indicating build completion. The maximum wait time for this operation is `CATKIN_BUILD_TIMEOUT_SECONDS`.

5. **ROS Service Verification**:
    - Confirm the health and operation of ROS services by checking for the service named `rosout` by running:

    ```bash
    docker exec container_name bash -c "source devel/setup.bash && rosservice list"
    ```

## Troubleshooting

If you encounter difficulties, the logs generated during the Docker-compose build and execution phases are your best first reference. These logs can provide insights into issues such as missing dependencies or other common errors.

Frequently observed challenges include:

- Launch command in Docker compose not set up correctly to match package name and launch file name.
- Failure to update the `CMakeList.txt` with the appropriate package name.
- Discrepancies between the package name specified in the launch command and those in `package.xml` or the launch file.
- Not updating the variables for the relevant ROS distribution in both the `Dockerfile` and `docker-entrypoint.sh`.
- Problems connecting to the Master node. Check the compose network bridge and the `ROS_MASTER_URI` variable within the docker-compose.
- The Master node isn't initiated prior to the package launch. Ensure the `--wait` flag is employed in the launch command.
- FS_msgs or other submodule not initialized. `git submodule update --init --recursive`.

## Resources

1. **ROS (Robot Operating System)**:
    - [ROS Introduction](http://wiki.ros.org/ROS/Introduction)
    - [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)
    - [roslaunch](http://wiki.ros.org/roslaunch)
    - [roslaunch XML format](http://wiki.ros.org/roslaunch/XML)
  
2. **Catkin Build System**:
    - [Catkin Introduction](http://wiki.ros.org/catkin)
    - [Catkin Tutorials](http://wiki.ros.org/catkin/Tutorials)
    - [Understanding CMakeLists.txt](http://wiki.ros.org/catkin/CMakeLists.txt)
    - [Understanding package.xml](http://wiki.ros.org/catkin/package.xml)

3. **Docker**:
    - [Docker Get Started Guide](https://docs.docker.com/get-started/)
    - [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
    - [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
  
4. **Docker Compose**:
    - [Overview of Docker Compose](https://docs.docker.com/compose/)
    - [Docker Compose CLI Reference](https://docs.docker.com/compose/reference/overview/)

5. **Github Actions**:

    - [Understanding Github Actions](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions)

