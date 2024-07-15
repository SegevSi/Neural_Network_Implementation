
# Neural Network Library

This repository contains a neural network library (`nnlibrary.cs`) implemented from scratch without using any machine learning or mathematical operation libraries. The library enables you to build custom neural network models, train and evaluate them.

In addition, the project contains an example (`Example.cs`) that demonstrates how to load and prepare data, build, train, and evaluate a neural network model that classifies handwritten digits using this library and the MNIST dataset.

## Features

- Build custom neural network models
- Train neural networks with custom data
- Evaluate the performance of trained models
- Supports various activation functions and loss functions
- Supports only one optimizer (SGD)
- Supports layer normalization and data preparation techniques

## Installation
To use this library, clone the repository to your local machine:

```bash
git clone https://github.com/SegevSi/Neural_Network_Implementation.git
cd neural-network-library
```

## Usage

### Running the Example

To run the `Example.cs` file, download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and follow the instructions in lines 72-73 of `Example.cs`:

```csharp
double[,] data_train = Read_CSV_File("C:\\Users\\hamra\\source\\repos\\Neural_Network_Implementation\\mnist_train.csv");  // change the parameter in Read_CSV_File to your path to mnist_train.csv 
double[,] data_test = Read_CSV_File("C:\\Users\\hamra\\source\\repos\\Neural_Network_Implementation\\mnist_test.csv");  // change the parameter in Read_CSV_File to your path to mnist_test.csv
```

Replace the paths with the appropriate paths to the `mnist_train.csv` and `mnist_test.csv` files on your local machine.

## Examples

The `Example.cs` file demonstrates how to:

- Load and prepare data
- Build a neural network model
- Train the model
- Evaluate the model's performance

This example specifically covers the classification of handwritten digits using the MNIST dataset and showcases the capabilities of the custom neural network library.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Ensure to replace the placeholder paths in the `Example.cs` file with the actual paths to the MNIST dataset on your local machine.
