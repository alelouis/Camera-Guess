//
//  utility_f.hpp
//  Neural Networks
//
//  Created by Alexis Louis on 17/03/2016.
//  Copyright © 2016 Alexis Louis. All rights reserved.
//

#ifndef utility_f_hpp
#define utility_f_hpp

#include "Header.h"
using namespace std;

vector<vector<float>> readMatFromFile(string str);
deque<deque<int>> readMatFromFileDeq(string str);
void write_mat_to_file(vector<vector<float>> vec, string name);
void write_vec_to_file(vector<float> vec, string name);

#endif /* utility_f_hpp */
