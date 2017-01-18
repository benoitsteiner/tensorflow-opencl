# TensorFlow for Java

Java bindings for TensorFlow.

> *WARNING*: The TensorFlow Java API is incomplete and experimental and can
> change without notice. Progress can be followed in
> [issue #5](https://github.com/tensorflow/tensorflow/issues/5).
>
> Till then, for using TensorFlow on Android refer to
> [contrib/android](https://www.tensorflow.org/code/tensorflow/contrib/android),
> [makefile](https://www.tensorflow.org/code/tensorflow/contrib/makefile#android)
> and/or the [Android camera
> demo](https://www.tensorflow.org/code/tensorflow/examples/android).

## Requirements

-   [bazel](https://www.bazel.build/versions/master/docs/install.html)
-   Environment to build TensorFlow from source code
    ([Linux](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-linux)
    or [Mac OS
    X](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-mac-os-x)).
    If you'd like to skip reading those details and do not care about GPU
    support, try the following:

    ```sh
    # On Linux
    sudo apt-get install python swig python-numpy

    # On Mac OS X with homebrew
    brew install swig
    ```

## Installation

Build the Java Archive (JAR) and native library:

```sh
bazel build -c opt \
  //tensorflow/java:tensorflow \
  //tensorflow/java:libtensorflow_jni
```

### Maven

To use the library in an external Java project, publish the library to a Maven
repository.  For example, publish the library to the local Maven repository
using the `mvn` tool (installed separately):

```sh
bazel build -c opt //tensorflow/java:pom
mvn install:install-file \
  -Dfile=../../bazel-bin/tensorflow/java/libtensorflow.jar \
  -DpomFile=../../bazel-bin/tensorflow/java/pom.xml
```

Refer to the library using Maven coordinates.  For example, if you're using
Maven then place this dependency into your `pom.xml` file (replacing
0.12.head with the version of the TensorFlow runtime you wish to use).

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>0.12.head</version>
</dependency>
```

## Example

### With bazel

Add a dependency on `//tensorflow/java:tensorflow` to the `java_binary` or
`java_library` rule. For example:

```sh
bazel run -c opt //tensorflow/java/src/main/java/org/tensorflow/examples:label_image
```

### With `javac`

-   Add `libtensorflow.jar` to classpath for compilation. For example:

    ```sh
    javac \
      -cp ../../bazel-bin/tensorflow/java/libtensorflow.jar \
      ./src/main/java/org/tensorflow/examples/LabelImage.java
    ```

-   Make `libtensorflow.jar` and `libtensorflow_jni.so`
    (`libtensorflow_jni.dylib` on OS X) available during execution. For example:

    ```sh
    java \
      -Djava.library.path=../../bazel-bin/tensorflow/java \
      -cp ../../bazel-bin/tensorflow/java/libtensorflow.jar:./src/main/java \
      org.tensorflow.examples.LabelImage
    ```
