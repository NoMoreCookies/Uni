#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Network {
public:
    int num_layers;
    std::vector<int> sizes;
    std::vector<VectorXd> biases;   // biases[l] – biasy dla warstwy l+1
    std::vector<MatrixXd> weights;  // weights[l] – wagi między warstwą l a l+1

    // konstruktor: sizes np. {2, 3, 1}
    explicit Network(const std::vector<int>& layer_sizes)
        : num_layers(static_cast<int>(layer_sizes.size())),
          sizes(layer_sizes)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);

        // biasy (dla wszystkich warstw oprócz wejściowej)
        for (int l = 1; l < num_layers; ++l) {
            int y = sizes[l];
            VectorXd b(y);
            for (int j = 0; j < y; ++j)
                b(j) = dist(gen);
            biases.push_back(b);

            int x = sizes[l - 1];
            MatrixXd w(y, x);
            for (int j = 0; j < y; ++j)
                for (int i = 0; i < x; ++i)
                    w(j, i) = dist(gen);
            weights.push_back(w);
        }
    }

    // forward pass: a_L = sigmoid(...sigmoid(Wx + b)...)
    VectorXd feedforward(const VectorXd& x) const {
        VectorXd a = x;
        for (std::size_t l = 0; l < biases.size(); ++l) {
            VectorXd z = weights[l] * a + biases[l];
            a = z.unaryExpr(&sigmoid);   // sigmoid element-wise
        }
        return a;
    }

    // proste SGD
    void SGD(std::vector<std::pair<VectorXd, VectorXd>>& training_data,
             int epochs, int mini_batch_size, double eta)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        int n = static_cast<int>(training_data.size());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(training_data.begin(), training_data.end(), gen);

            for (int k = 0; k < n; k += mini_batch_size) {
                int end = std::min(n, k + mini_batch_size);
                std::vector<std::pair<VectorXd, VectorXd>> mini_batch(
                    training_data.begin() + k,
                    training_data.begin() + end
                );
                update_mini_batch(mini_batch, eta);
            }

            std::cout << "Epoch " << (epoch + 1) << " complete\n";
        }
    }

private:
    // --- aktualizacja wag/biasów na podstawie jednego mini-batcha ---
    void update_mini_batch(const std::vector<std::pair<VectorXd, VectorXd>>& mini_batch,
                           double eta)
    {
        // gradienty sumowane po próbkach
        std::vector<VectorXd> nabla_b;
        std::vector<MatrixXd> nabla_w;
        nabla_b.reserve(biases.size());
        nabla_w.reserve(weights.size());

        for (std::size_t l = 0; l < biases.size(); ++l) {
            nabla_b.emplace_back(VectorXd::Zero(biases[l].size()));
            nabla_w.emplace_back(MatrixXd::Zero(weights[l].rows(), weights[l].cols()));
        }

        // każda próbka → backprop → dodajemy do nabla_b, nabla_w
        for (const auto& sample : mini_batch) {
            const VectorXd& x = sample.first;
            const VectorXd& y = sample.second;

            std::vector<VectorXd> delta_nabla_b(biases.size());
            std::vector<MatrixXd> delta_nabla_w(weights.size());
            for (std::size_t l = 0; l < biases.size(); ++l) {
                delta_nabla_b[l] = VectorXd::Zero(biases[l].size());
                delta_nabla_w[l] = MatrixXd::Zero(weights[l].rows(), weights[l].cols());
            }

            backprop(x, y, delta_nabla_b, delta_nabla_w);

            for (std::size_t l = 0; l < biases.size(); ++l) {
                nabla_b[l] += delta_nabla_b[l];
                nabla_w[l] += delta_nabla_w[l];
            }
        }

        double m = static_cast<double>(mini_batch.size());
        for (std::size_t l = 0; l < biases.size(); ++l) {
            biases[l] -= (eta / m) * nabla_b[l];
            weights[l] -= (eta / m) * nabla_w[l];
        }
    }

    // --- backprop dla pojedynczej próbki (x,y) ---
    void backprop(const VectorXd& x, const VectorXd& y,
                  std::vector<VectorXd>& nabla_b,
                  std::vector<MatrixXd>& nabla_w) const
    {
        // 1. forward: zapisujemy wszystkie z^l i a^l
        VectorXd activation = x;
        std::vector<VectorXd> activations;  // a^0, a^1, ..., a^L
        std::vector<VectorXd> zs;           // z^1, ..., z^L

        activations.push_back(activation);

        for (std::size_t l = 0; l < biases.size(); ++l) {
            VectorXd z = weights[l] * activation + biases[l];
            zs.push_back(z);
            activation = z.unaryExpr(&sigmoid);
            activations.push_back(activation);
        }

        // 2. wyjściowa delta_L = (a^L - y) ⊙ σ'(z^L)
        VectorXd delta = cost_derivative(activations.back(), y)
                         .cwiseProduct(zs.back().unaryExpr(&sigmoid_prime));

        int last = static_cast<int>(biases.size()) - 1;
        nabla_b[last] = delta;
        nabla_w[last] = delta * activations[activations.size() - 2].transpose();

        // 3. warstwy ukryte: delta^l = (W^{l+1}^T delta^{l+1}) ⊙ σ'(z^l)
        for (int l = last - 1; l >= 0; --l) {
            const VectorXd& z = zs[static_cast<std::size_t>(l)];
            VectorXd sp = z.unaryExpr(&sigmoid_prime);
            delta = (weights[static_cast<std::size_t>(l) + 1].transpose() * delta)
                    .cwiseProduct(sp);

            nabla_b[static_cast<std::size_t>(l)] = delta;
            nabla_w[static_cast<std::size_t>(l)] =
                delta * activations[static_cast<std::size_t>(l)].transpose();
        }
    }

    // --- koszt kwadratowy: pochodna po a^L ---
    static VectorXd cost_derivative(const VectorXd& output_activations,
                                    const VectorXd& y)
    {
        return output_activations - y;   // a^L - y
    }

    // --- sigmoid + pochodna ---
    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    static double sigmoid_prime(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }
};


// --- przykład użycia: XOR ---

int main() {
    Network net({2, 3, 1});

    std::vector<std::pair<VectorXd, VectorXd>> training_data;

    auto make_sample = [](double x1, double x2, double y) {
        VectorXd in(2);
        in << x1, x2;
        VectorXd out(1);
        out << y;
        return std::make_pair(in, out);
    };

    training_data.push_back(make_sample(0.0, 0.0, 0.0));
    training_data.push_back(make_sample(0.0, 1.0, 1.0));
    training_data.push_back(make_sample(1.0, 0.0, 1.0));
    training_data.push_back(make_sample(1.0, 1.0, 0.0));

    net.SGD(training_data, /*epochs=*/10000, /*mini_batch_size=*/4, /*eta=*/0.5);

    for (const auto& sample : training_data) {
        VectorXd out = net.feedforward(sample.first);
        std::cout << sample.first.transpose()
                  << " -> " << out(0) << "\n";
    }
}
