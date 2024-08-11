#include <iostream>
#include <fstream>
#include <vector>
#include <string>


std::vector<double> read_predictions(const std::string& filename) {
    std::vector<double> predictions;
    std::ifstream file(filename);
    double value;
    while (file >> value) {
        predictions.push_back(value);
    }
    return predictions;
}

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
    std::vector<double> predictions = read_predictions("predictions.txt");
    trading_strategy(predictions, 0.01);
    return 0;
}
