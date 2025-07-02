#include "annslib.h"

int
main() {
    CsvLogger logger("my_experiment_results.csv", 4);

    if (!logger.isOpen()) {
        std::cerr << "Failed to initialize CsvLogger. Exiting." << std::endl;
        return 1;
    }

    std::vector<std::string> headers = {
        "TrialID", "Parameter_X", "Parameter_Y", "Result_Value", "Status", "Notes"};
    if (!logger.writeHeader(headers)) {
        return 1;
    }
    std::cout << "CSV Header written." << std::endl;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 100.0);

    for (int i = 1; i <= 5; ++i) {
        int trial_id = i;
        double param_x = distribution(generator);
        double param_y = distribution(generator) / 2.0;
        double result = param_x * 0.5 + param_y * 1.2 + i * 0.1;
        std::string status = (result > 70.0) ? "PASS" : "FAIL";

        std::string notes;
        if (i == 2) {
            notes = "This note has a comma, and a \"quote\" inside.";
        } else if (i == 4) {
            notes = "Another test note.\nWith a new line character.";
        } else {
            notes = "Normal run";
        }

        logger.writeRow(std::vector<std::string>{std::to_string(trial_id),  // int 转换为 string
                                                 std::to_string(param_x),  // double 转换为 string
                                                 std::to_string(param_y),
                                                 std::to_string(result),
                                                 status,
                                                 notes});

        logger.writeRow(std::vector<double>{
            (double)trial_id,  // 整数转换为double
            param_x,
            param_y,
            result,
        });

        std::cout << "Data for trial " << i << " written." << std::endl;
    }

    std::cout << "Experiment finished. Data saved to my_experiment_results.csv" << std::endl;

    return 0;
}