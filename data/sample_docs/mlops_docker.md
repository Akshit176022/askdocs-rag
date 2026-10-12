# MLOps and Docker

Docker packages an application with its dependencies into an image. A container is a running instance of an image, which makes software behave the same on a laptop and on a server.

A Dockerfile lists the build steps. Each instruction creates a layer, and Docker caches unchanged layers, so copying the requirements file and installing dependencies before copying the source code makes rebuilds faster.

Experiment tracking tools such as MLflow or Weights and Biases record parameters, metrics and artifacts for each run, so results can be compared and reproduced.

Models are often served behind a REST API. FastAPI is a Python framework that builds such APIs and validates request bodies with Pydantic models.

To reduce latency and cost, teams cache repeated computations, batch requests, and shrink models with quantization, which stores weights in lower precision such as 8-bit integers.
