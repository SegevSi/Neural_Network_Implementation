using System;
using System.IO;
namespace neural_network_implementation
{
    public interface IFunc
    {
        public double Func(double x);
        public double Derivative(double x);


    }
    public interface ICost
    {
        public double Cost_Func(double x, double y);
        public double Derivative(double x, double y, int n);
    }
    public class Sigmoind : IFunc
    {
        public double Func(double x)
        {
            return 1 / (1 + Math.Exp(-x));

        }
        public double Derivative(double x)
        {
            double y = Func(x);
            return y * (1 - y);
        }
    }
    public class Mean_Squre_Erorrs : ICost
    {
        public double Cost_Func(double x, double y)
        {
            double z = x - y;
            return z * z;
        }
        public double Derivative(double x, double y, int n)
        {
            return 2 * (x - y) / n;
        }
    }
    public class RelU : IFunc
    {
        public double Func(double x)
        {
            if (x < 0) return 0;
            return x;
        }
        public double Derivative(double x)
        {
            if (x > 0) return 1;
            if (x < 0) return 0;
            throw new Exception("u cant put 0 in the derivative of relu ");
        }
    }
    public class Linear : IFunc
    {
        public double Func(double x) { return x; }
        public double Derivative(double x) { return 1; }
    }

    class Program
    {

        static double[,] Read_CSV_File(string file_name)
        {
            string csvFilePath = file_name;

            // Check if the CSV file exists
            if (File.Exists(csvFilePath))
            {
                // Read the CSV file into a matrix of double
                double[,] matrix;

                using (StreamReader reader = new StreamReader(csvFilePath))
                {
                    // Read the first line to determine the number of columns
                    string[] headers = reader.ReadLine().Split(',');
                    int colCount = headers.Length;

                    // Count the number of data rows
                    int rowCount = 1; // Start from 1 to account for the header row
                    while (!reader.EndOfStream)
                    {
                        reader.ReadLine();
                        rowCount++;
                    }

                    // Initialize the matrix
                    matrix = new double[rowCount, colCount];

                    // Reset the reader to the beginning of the file
                    reader.DiscardBufferedData();
                    reader.BaseStream.Seek(0, SeekOrigin.Begin);

                    // Read and parse the data into the matrix
                    for (int row = 0; row < rowCount; row++)
                    {
                        string[] rows = reader.ReadLine().Split(',');
                        for (int col = 0; col < colCount; col++)
                        {
                            if (double.TryParse(rows[col], out double value))
                            {
                                matrix[row, col] = value;
                            }
                            else
                            {
                                // Handle cases where the cell doesn't contain a valid double value
                                matrix[row, col] = 0.0; // or another default value
                            }
                        }
                    }
                }

                return matrix;
            }
            else
            {
                throw new InvalidDataException("the csv file does not exsist");

            }
        }


        static void Main(string[] args)
        {
            double[,] data_train = Read_CSV_File("C:\\Users\\hamra\\Desktop\\אקסל לניתוח מידע\\mnist_train.csv"); // path to your mnist_train.csv file 
            double[,] data_test = Read_CSV_File("C:\\Users\\hamra\\Desktop\\אקסל לניתוח מידע\\mnist_test.csv"); // path to your mnist_test.csv file 
            (double[,] x, int[] y) divided_data_train = CLC.Pull_Column(data_train, 0);
            (double[,] x, int[] y) divided_data_test = CLC.Pull_Column(data_test, 0);
            double[,] x_train = divided_data_train.x;
            double[,] y_train = CLC.One_hot_encoder(10, divided_data_train.y);
            double[,] x_test = divided_data_test.x;
            double[,] y_test = CLC.One_hot_encoder(10, divided_data_test.y);
            NN model = new NN();
            model.Add(new Input_Layer(28 * 28));
            model.Add(new Layer(128, new Sigmoind(), layer_norm: true));
            model.Add(new Layer(64, new Sigmoind(), layer_norm: true));
            model.Add(new Layer(10, new Sigmoind()));
            model.Init_W(-1, 1);
            model.Compile(0.1, 5, x_train, y_train, x_test, y_test, new Mean_Squre_Erorrs());
            model.Fit_W_Sparse_Categorical_Accuracy(64); // SGD is the  optimizer
            double[,] input =
           {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
            double[] flat_input = Matrix.Flatten(input);
            double[] result = model.Predict1(flat_input);
            Arr.PrintDoubleArray(result);
            Console.WriteLine("the number is " + Arr.Max_index(result));

        }
    }


}
