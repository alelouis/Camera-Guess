//
//  utility_f.cpp
//  Neural Networks
//
//  Created by Alexis Louis on 17/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#include "utility_f.hpp"

vector<vector<float>> readMatFromFile(string str){
    vector<vector<float>> matAPP;
    vector<float> lineAPP;
    string line;;
    ifstream APP;
    APP.open(str, ios::in);

    if (!APP)
        std::cout << "Cannot open file \"" << str << "\"" << std::endl;
    else
    {
        std::vector<int> v;

        while (std::getline(APP, line))
        {
            std::istringstream iss(line);
            float n;
            while (iss >> n)
            {
                lineAPP.push_back(n);
            }
            matAPP.push_back(lineAPP);
            lineAPP.clear();
        }
    }

    return matAPP;
}

deque<deque<int>> readMatFromFileDeq(string str){
    deque<deque<int>> matAPP;
    deque<int> lineAPP;
    string line;
    ifstream APP;
    APP.open(str, ios::in);
    std::deque<int> v;

    while (std::getline(APP, line))
    {
        std::istringstream iss(line);
        int n;
        while (iss >> n)
        {
            lineAPP.push_back(n);
        }
        matAPP.push_back(lineAPP);
        lineAPP.clear();
    }

    return matAPP;
}

void write_mat_to_file(vector<vector<float>> vec, string name){
    ofstream myfile;
    myfile.open (name);
    for(int i=0;i<vec.size();i++){
        for(int j=0;j<vec[0].size();j++){
            myfile << vec[i][j];
            myfile << " ";
        }
        myfile << '\n';
    }
    myfile.close();
}

void write_vec_to_file(vector<float> vec, string name){
    ofstream myfile;
    myfile.open (name);
    for(int i=0;i<vec.size();i++){
            myfile << vec[i];
            myfile << " ";
        }
    myfile.close();
}

