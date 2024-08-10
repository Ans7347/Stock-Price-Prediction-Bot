#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Function to read predictions from a file
std::vector<double> read_predictions(const std::string& filename) {
    std::vector<double> predictions;
    std::ifstream file(filename);
    double value;
    while (file >> value) {
        predictions.push_back(value);
    }
    return predictions;
}

// Simple trading strategy
void trading_strategy(const std::vector<double>& predictions, double threshold) {
    double last_prediction = predictions[0];
    for (size_t i = 1; i < predictions.size(); ++i) {
        if (predictions[i] > last_prediction + threshold) {
            std::cout << "Buy at prediction " << predictions[i] << std::endl;
        } else if (predictions[i] < last_prediction - threshold) {
            std::cout << "Sell at prediction " << predictions[i] << std::endl;
        }
        last_prediction = predictions[i];
    }
}

int main() {
    // Load predictions made by the TensorFlow model
    std::vector<double> predictions = read_predictions("predictions.txt");

    // Execute the trading strategy
    trading_strategy(predictions, 0.01);

    return 0;
}
