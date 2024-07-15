using System;
using System.Collections.Generic;
using System.Text;

namespace neural_network_implementation
{
    public class Neuron
    {
        public readonly double[] net;
        public readonly double[] _out;
        public readonly double[] derevtive;

        public Neuron(double[] net, IFunc func)
        {
            this.net = net;
            this._out = new double[this.net.Length];
            this.derevtive = new double[this.net.Length];
            for (int i = 0; i < _out.Length; i++)
            {
                this._out[i] = func.Func(this.net[i]);
                this.derevtive[i] = func.Derivative(this.net[i]);
            }
        }

        public Neuron(double[] outer)
        {
            this._out = outer;
            this.net = null;
        }
    }



    public class Arr
    {
        private double[] arr;
        public Arr(int l)
        {
            arr = new double[l];
        }
        public Arr(double[] arr)
        {
            this.arr = arr;

        }
        

        public static double[] operator +(Arr other1, double[] arr)
        {
            if (other1.arr.Length != arr.Length) throw new InvalidOperationException(" the arrays are not in the same size, therefore u cant add them together");
            for (int i = 0; i < arr.Length; i++)
                arr[i] += other1.arr[i];
            return arr;
        }
        public void Update(double[] x, double lr)
        {
            for (int i = 0; i < arr.Length; i++)
                this.arr[i] -= x[i] * lr;

        }

        public static double Avg(double[] x)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i];
            }

            return sum / x.Length;
        }

        public static double Standard_deviation(double[] x)
        {
            double avg = Arr.Avg(x);
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += Math.Pow(x[i] - avg, 2);
            }
            sum = sum / x.Length;
            return Math.Sqrt(sum);
        }

        public static int Max_index(double[] arr)
        {
            int max_index = 0;
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[max_index] < arr[i])
                    max_index = i;
            }
            
            return max_index;

        }

        public static void PrintDoubleArray(double[] array)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array), "Array cannot be null.");
            }

            Console.Write("[ ");

           
            for (int i = 0; i < array.Length; i++)
            {
                
                Console.Write(array[i]);

                
                if (i < array.Length - 1)
                {
                    Console.Write(", ");
                }
            }

            
            Console.WriteLine(" ]");
        }
    }
    public class Matrix
    {

        private double[,] mat;
        public Matrix(double[,] m)
        {
            mat = m;
        }
        public Matrix(int row, int col)
        {
            mat = new double[row, col];
            

        }

        public void Random(int  min, int  max)
        {
            Random x = new Random();
            for (int i = 0; i < this.mat.GetLength(0); i++)
            {
                for (int g = 0; g < this.mat.GetLength(1); g++)
                {
                    int y = x.Next(min, max + 1);
                    if (y > 0)
                        this.mat[i, g] = x.NextDouble() * (-1) + y;
                    else this.mat[i, g] = x.NextDouble() + y;
                }
            }
        }


        public static double[] operator *(Matrix other1, double[] x)// horizontaly
        {
            if (other1.mat.GetLength(1) != x.Length) throw new InvalidOperationException(" the number of the columns of the matrix  is  not equal to the length of the array , therefore u cant * them together");

            double[] n = new double[other1.mat.GetLength(0)];
            for (int i = 0; i < other1.mat.GetLength(0); i++)
            {
                for (int g = 0; g < x.Length; g++)
                    n[i] += other1.mat[i, g] * x[g];
            }

            return n;
        }

        
        public void Update(double[,] x, double lr)
        {
            for (int i = 0; i < this.mat.GetLength(0); i++)
            {
                for (int g = 0; g < this.mat.GetLength(1); g++)
                {
                    this.mat[i, g] -= x[i, g] * lr;
                }
            }

        }
        public static double[] Flatten(double[,] mat)
        {
            double[] arr = new double[mat.Length];
            int count = 0;
            for (int i = 0; i < mat.GetLength(0); i++)
            {
                for (int g = 0; g < mat.GetLength(1); g++)
                {
                    arr[count] = mat[i, g];
                    count++;
                }
            }
            return arr;
        }
        public double[] Multiply_Verticly(double[] arr)
        {
            if (arr.Length != this.mat.GetLength(0)) throw new ArgumentException();
            double[] x = new double[this.mat.GetLength(1)];
            for (int i = 0; i < x.Length; i++)
            {
                for (int g = 0; g < arr.Length; g++)
                {
                    x[i] += this.mat[g, i] * arr[g];
                }
            }


            return x;

        }


    }

    public class Layer
    {
        private IFunc func;
        private Matrix weight;
        private Arr bias;
        private int input_size;
        private int output_size;
        public readonly bool do_layer_norm;
        public readonly bool built_with_input_size;
        private bool can_set_input_size = true ;
        public Layer(int input_size, int output_size, IFunc x, bool layer_norm = false)
        {
            this.input_size = input_size;
            this.output_size = output_size;
            this.weight = new Matrix(new double[output_size, input_size]);
            this.bias = new Arr(this.output_size);
            this.func = x;
            this.do_layer_norm = layer_norm;
            this.built_with_input_size = true;
            this.can_set_input_size = false;
        }
        public Layer(int output_size, IFunc x, bool layer_norm = false) // constructor that doesnt take input size 
        {

            this.output_size = output_size;
            this.bias = new Arr(this.output_size);
            this.func = x;
            this.do_layer_norm = layer_norm;
            this.built_with_input_size = false;
        }
        public void Set_Input_Size(int size)
        {
            if (this.can_set_input_size )
            {
                this.input_size = size;
                this.weight = new Matrix(new double[this.output_size, this.input_size]);
                this.can_set_input_size = false;
            }
            else throw new Exception("cant set input size more than one time ");

        }
        public void Random_ws(int  min, int  max)
        {
            this.weight.Random(min, max);

        }
        public int Get_Input_Size()
        { return this.input_size; }
        public int Get_Output_Size() { return this.output_size; }
        public Neuron Predict(double[] n)
        {
            if (this.input_size != n.Length) throw new ArgumentException("Layer.Predict function got an array that has the wrong length "); 
            double[] x = this.weight * n;
            if (!this.do_layer_norm) return new Neuron(this.bias + x, this.func);
            return new Neuron(this.Layer_norm(this.bias + x), this.func);
        }


        public void Update(double[,] w_h, double[] b_h, double lr)
        {
            this.weight.Update(w_h, lr);
            this.bias.Update(b_h, lr);

        }

        public double[] Multiply_weight_Verticly(double[] arr)
        {
            return weight.Multiply_Verticly(arr);
        }
        private double[] Layer_norm(double[] x)
        {
            double avg = Arr.Avg(x);
            double sd = Arr.Standard_deviation(x);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (x[i] - avg) / sd;
            }
            return x;
        }
    }
    public class Input_Layer
    {
        public readonly int size;
        public Input_Layer(int size)
        {
            this.size = size;
        }
    }
    public class BackpropagationSaver
    {
        public double[,] w;
        public double[] b;
        public BackpropagationSaver(int size, int next)
        {
            w = new double[next, size];
            b = new double[next];
        }

        public void Add(double[,] w, double[] b)
        {
            for (int i = 0; i < this.b.Length; i++)
            {
                this.b[i] += b[i];
                for (int g = 0; g < this.w.GetLength(1); g++)
                {
                    this.w[i, g] += w[i, g];
                }

            }



        }
    }



    public class NN
    {

        private readonly List<Layer> layers;
        private double lr;
        private int epoches; 
        private double[,] x_train;
        private double[,] y_train;
        private double[,] x_test;
        private double[,] y_test;
        public double[] Costs_per_Epoche; 
        private ICost cost_func;
        private List<BackpropagationSaver> BackpropagationSavers;
        private int input_layer_size;
        public NN()
        {
            this.layers = new List<Layer>();

        }
        public void Add(object x)
        {
            if (x is Input_Layer)
                this.input_layer_size = ((Input_Layer)x).size;

            else if (x is Layer)
            {
                Layer l = (Layer)x;
                if (l.built_with_input_size)
                    this.layers.Add(l);
                else
                {
                    if (this.layers.Count == 0)
                    {
                        if (input_layer_size == 0)
                            throw new Exception();
                        else l.Set_Input_Size(input_layer_size);
                    }
                    else
                    {
                        l.Set_Input_Size(this.layers[this.layers.Count - 1].Get_Output_Size());
                    }

                }
                this.layers.Add(l);
            }

        }
        private void Check_Layers() // check if the layers can be a neural network  
        {
            
            for (int i = 0; i < this.layers.Count - 1; i++)
            {
                int size1 = this.layers[i].Get_Input_Size();
                int size2 = this.layers[i + 1].Get_Input_Size();
                int next1 = this.layers[i].Get_Output_Size();
                int next2 = this.layers[i + 1].Get_Output_Size();
                if (size1 == 0 || size2 == 0 || next1 == 0 || next2 == 0 || next1 != size2)
                    throw new Exception("the layers were not build corectly ");
            }


        }
        private double[] Pull_Row(double[,] mat, int row_num)
        {
            if (row_num >= mat.GetLength(0) || row_num < 0) throw new IndexOutOfRangeException();
            double[] arr = new double[mat.GetLength(1)];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = mat[row_num, i];
            return arr;
        }
        public double[] Predict1(double[] input)
        {
            double[] x = input;
            for (int i = 0; i < this.layers.Count; i++)
            {
                x = this.layers[i].Predict(x)._out;
            }
            return x;
        }
        private List<Neuron> Predict1_4training(double[] input)
        {
            List<Neuron> ns = new List<Neuron>();
            ns.Add(new Neuron(input));
            double[] x = input;
            for (int i = 0; i < this.layers.Count; i++)
            {
                Neuron n = this.layers[i].Predict(x);
                ns.Add(n);
                x = n._out;
            }
            return ns;
        }
        public double[][] Predict(double[,] input)  
        {
            double[][] outcomes = new double[input.GetLength(0)][];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                double[] arr = this.Pull_Row(input, i);
                outcomes[i] = this.Predict1(arr);

            }
            return outcomes;

        }
        private double Cost_func(double[,] x, double[,] y)
        {
            double sum = 0;
            double[][] y_pred = Predict(x);
            int size = y.Length;
            for (int i = 0; i < y.GetLength(0); i++)
            {

                for (int g = 0; g < y.GetLength(1); g++)
                {

                    sum += cost_func.Cost_Func(y_pred[i][g], y[i, g]);
                }
            }
            return sum / size;
        }
        public void Init_W(int min, int max)
        {
            for (int i = 0; i < this.layers.Count; i++)
                this.layers[i].Random_ws(min, max);


        }

        public void Compile(double lr, int epoches, double[,] x_train, double[,] y_train, double[,] x_test, double[,] y_test, ICost cost_func)
        {
            this.lr = lr;
            this.epoches = epoches;
            this.x_train = x_train;
            this.y_train = y_train;
            this.x_test = x_test;
            this.y_test = y_test;
            this.Costs_per_Epoche = new double[this.epoches];
            this.cost_func = cost_func;
            int input = this.layers[0].Get_Input_Size();
            int output = this.layers[this.layers.Count - 1].Get_Output_Size();
            if (input != this.x_train.GetLength(1) || input != this.x_test.GetLength(1) || output != this.y_train.GetLength(1) || output != this.y_test.GetLength(1))
                throw new Exception("the data is not suiteble for the NN architucture ");

            if (this.x_train.GetLength(0) != this.y_train.GetLength(0) && this.x_test.GetLength(0) != this.y_test.GetLength(0))
                throw new Exception("the training and the testing  data dont have the same samples (between x and y ) ");

            if (this.x_train.GetLength(0) != this.y_train.GetLength(0))
                throw new Exception("the training  data dont have the same samples (between x and y ) ");

            if (this.x_test.GetLength(0) != this.y_test.GetLength(0))
                throw new Exception("the testing  data dont have the same samples (between x and y ) ");
        }

        public void Update()
        {
            for (int i = 0; i < this.layers.Count; i++)
            {
                this.layers[i].Update(this.BackpropagationSavers[i].w, this.BackpropagationSavers[i].b, this.lr);
            }


        }
        public void Init_BackpropagationSavers_List()
        {
            this.BackpropagationSavers = new List<BackpropagationSaver>();
            for (int i = 0; i < this.layers.Count; i++)
            {
                this.BackpropagationSavers.Add(new BackpropagationSaver(this.layers[i].Get_Input_Size(), this.layers[i].Get_Output_Size()));
            }
        }
        private double[] MultiplyArrs(double[] arr1, double[] arr2)
        {
            double[] arr = new double[arr1.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = arr2[i] * arr1[i];
            }
            return arr;

        }
        private double[,] D_for_weights(double[] d_of_net_after_ws, double[] out_layer_before_ws)
        {
            double[,] d_weights = new double[d_of_net_after_ws.Length, out_layer_before_ws.Length];
            for (int i = 0; i < d_of_net_after_ws.Length; i++)
            {
                for (int g = 0; g < out_layer_before_ws.Length; g++)
                {
                    d_weights[i, g] = d_of_net_after_ws[i] * out_layer_before_ws[g];
                }
            }

            return d_weights;
        }
        private void Add_BP_To_BackpropagationSavers(double[] x, double[] y, int batch_size)
        {
            List<Neuron> layers_val = this.Predict1_4training(x);
            double[] y_pred = layers_val[layers_val.Count - 1]._out;
            double[] dcosts = new double[y_pred.Length];
            for (int i = 0; i < y_pred.Length; i++)
            {
                dcosts[i] = this.cost_func.Derivative(y_pred[i], y[i], y.Length) / batch_size;

            }
            double[] d_of_out = dcosts;

            for (int i = layers_val.Count - 1; i > 0; i--)
            {
                double[] d_biases;
                double[]  d_net  = this.MultiplyArrs(d_of_out, layers_val[i].derevtive);
                d_biases = d_net;
                double[,] d_weights = this.D_for_weights(d_net, layers_val[i - 1]._out);
                this.BackpropagationSavers[i - 1].Add(d_weights, d_biases);
                if (i != 1)
                    d_of_out = this.layers[i - 1].Multiply_weight_Verticly(d_net);



            }


        }
        


        private void One_Epoch(int batch_size)
        {

            int n = this.x_train.GetLength(0);
            for (int i = 0; i < n; i += batch_size)
            {
                this.Init_BackpropagationSavers_List();
                for (int g = i; g < g + batch_size && g < n; g++)
                {
                    this.Add_BP_To_BackpropagationSavers(Pull_Row(this.x_train, g), Pull_Row(this.y_train, g), batch_size);
                }
                this.Update();
            }
        }




        public double Accurecy_for_clc(double[,] x__test, double[,] y__test)
        {
            return CLC.SparseCategoricalAccuracy(this.Predict(x__test), y__test);

        }



        public void Fit(int batch_size)
        {
            this.Check_Layers();
            for (int i = 0; i < this.epoches; i++)
            {
                this.One_Epoch(batch_size);
                double cost = this.Cost_func(this.x_test, this.y_test);
                this.Costs_per_Epoche[i] = cost;
                Console.WriteLine($"epoch : {i + 1} cost : { cost}");
            }
        }

        public void Fit_W_Sparse_Categorical_Accuracy(int batch_size)
        {
            this.Check_Layers();
            for (int i = 0; i < this.epoches; i++)
            {
                this.One_Epoch(batch_size);
                double cost = this.Cost_func(this.x_test, this.y_test);
                double accuracy = this.Accurecy_for_clc(this.x_test, this.y_test);
                this.Costs_per_Epoche[i] = cost;
                Console.WriteLine($"epoch : {i + 1} cost : { cost} accuracy : {accuracy }");
            }
        }
    }

    
    public class CLC
    {
        public static double[,] One_hot_encoder(int num_of_classes, int[] y)
        {
            if (num_of_classes <= 0) throw new ArgumentException("the number of classes cant be <=0");
            double[,] normalised_y = new double[y.Length, num_of_classes];
            for (int i = 0; i < y.Length; i++)
            {   if(y[i]<0 || y[i]>=num_of_classes) throw new ArgumentException($"{nameof(y)} in index {i} can not  be {y[i]}");
                normalised_y[i, y[i]] = 1;
            }

            return normalised_y;
        }

        public static (double[,] x, int[] y) Pull_Column(double[,] mat, int col_num)
        {
            if (col_num >= mat.GetLength(1) || col_num < 0) throw new ArgumentException("the number of the column doesnt exist in the matrix ");
            int[] col = new int[mat.GetLength(0)];
            double[,] x = new double[mat.GetLength(0), mat.GetLength(1) - 1];

            for (int i = 0; i < mat.GetLength(0); i++)
            {
                int count = 0;
                for (int g = 0; g < mat.GetLength(1); g++)
                {
                    if (g == col_num)
                    {
                        col[i] = (int)mat[i, g];
                        
                    }
                    else
                    {
                        x[i, count] = mat[i, g];
                        count++;
                    }
                }
            }
            return (x, col);
        }
        public static double SparseCategoricalAccuracy(double[][] y_pred, double[,] y)
        {
            double sum = 0;
            int n = y.GetLength(0);
            for (int i = 0; i < y.GetLength(0); i++)
            {
                int index_pred = 0;
                double max_pred = y_pred[i][0];
                int index = 0;
                double max = y[i, 0];
                for (int g = 1; g < y.GetLength(1); g++)
                {
                    if (max < y[i, g])
                    {
                        max = y[i, g];
                        index = g;
                    }
                    if (max_pred < y_pred[i][g])
                    {
                        max_pred = y_pred[i][g];
                        index_pred = g;
                    }
                }
                if (index == index_pred)
                    sum++;
            }

            return sum / n;
        }



    }

}
