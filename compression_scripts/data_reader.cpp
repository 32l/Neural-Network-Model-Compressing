#include <cmath>
// #include <direct.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


using namespace std;

enum Layer_type { SQUEEZE3x3, OTHER };
enum Param_type { WEIGHT, BIAS};

int get_wb_stored_num(const int count, Layer_type lt) {
  if (lt == SQUEEZE3x3) {
    return (count - 1) / 3 + 1;
  } else {
    return (count - 1) / 4 + 1;
  }
}


void weight_check(std::vector<std::vector<unsigned short> > wb, int layer_id){
  std::vector<unsigned short> v_s = wb[layer_id -1];
  int size = v_s.size();
  int num_to_print = 10;
  char *space = "     ";
  cout << "---- total short: " << size <<":"<< endl;
  if (size <= 2* num_to_print) {
    printf("%s", space);
    for(int i =0; i< size; ++i){
      printf("%04x ", v_s[i]);
      if ( (i+1)%10 ==0) {
        printf("\n%s", space);
      }
    }
    printf("\n");
  } else {
    cout << "---- first "<< num_to_print <<":" <<endl;
    printf("%s", space);
    for ( int i =0; i< num_to_print; ++i){
      printf("%04x ", v_s[i]);
    }
    printf("\n     ......\n");
    cout << "---- last " << num_to_print <<": "<< endl;
    printf("%s", space);
    int r_start = size - num_to_print;
    for (int i = r_start; i< size; ++i) {
      printf("%04x ", v_s[i]);
    }
    printf("\n");
  }
}


std::map<unsigned char, float> get_code_book(const char min, const int bits) {
  std::map<unsigned char, float> code_book;

  for (unsigned char i = 0; i < (1 << (bits - 1)) - 1; ++i) {
    code_book.insert(pair<unsigned char, float>(i, -pow(2.0, min + i)));
  }
  code_book.insert(pair<unsigned char, float>((1 << (bits - 1)) - 1, 0.0));

  unsigned char mid = 1 << (bits - 1); // 8
  for (unsigned char i = mid; i < (1 << bits) - 1; ++i) {
    code_book.insert(pair<unsigned char, float>(i, pow(2.0, min + i - mid)));
  }
  // output code_book for checking
  map<unsigned char, float>::iterator iter;
  for (iter = code_book.begin(); iter != code_book.end(); ++iter) {
    printf("---- %x, %f \n", iter->first, iter->second);
  }
  return code_book;
}


void read_half_layer(ifstream &in, int layer_id, vector<char> &min_exp_w,
                     vector<int> &count_w,
                     vector<vector<unsigned short> > &weights_s,
                     vector<vector<float> > &weights, Param_type PT) {

  int init_exp = -100;
  char tmp_c;
  int tmp_i;
  shared_ptr<unsigned short> tmp_short_ptr;
  // read the minimum exp for this block of data
  in.read(&tmp_c, sizeof(char));
  printf("-- min_exp = %d ...\n", tmp_c);
  if (tmp_c > 0 ) {
    printf("Error: mim_exp might be too large! \n");
    exit(-1);
  }
  min_exp_w.push_back(tmp_c);
  cout << "-- code_book:" << endl;
  map<unsigned char, float> code_book;
  code_book = get_code_book(tmp_c, 4);
  // read the total number of data
  in.read(reinterpret_cast<char *>(&tmp_i), sizeof(int));
  cout << "-- param count: " << tmp_i << endl;
  count_w.push_back(tmp_i);

  // conv layers' & squeeze 3x3 layers' weights
  // if ((layer_id - 1) % 3 == 0 && layer_id != 1 && PT == WEIGHT) {
  if (((layer_id - 1) % 3 == 0 || layer_id == 26) && PT == WEIGHT) {
    // read the short
    int short_len = get_wb_stored_num(count_w[layer_id - 1], SQUEEZE3x3);
    cout << "-- 3x3 weights short_len = " << short_len << endl;
    tmp_short_ptr.reset(new unsigned short[short_len]);
    in.read(reinterpret_cast<char *>(tmp_short_ptr.get()),
            short_len * sizeof(unsigned short) / sizeof(char));
    std::vector<unsigned short> w_short_tmp;
    for (int i = 0; i < short_len; ++i) {
      w_short_tmp.push_back(tmp_short_ptr.get()[i]);
      // weights_s[layer_id - 1].push_back(tmp_short_ptr.get()[i]);
    }
    weights_s.push_back(w_short_tmp);
    cout<< "-- short formatted weights or bias: " << endl;
    weight_check(weights_s, layer_id);
    // store the real values to weights

    std::vector<float> w_float_tmp;
    for (int i = 0; i < short_len; ++i) {
      unsigned short tmp_short = weights_s[layer_id - 1][i];
      // first
      unsigned char code = 0xF & tmp_short;
      w_float_tmp.push_back(code_book.find(code)->second);
      // second
      code = 0xF & (tmp_short >> 4);
      w_float_tmp.push_back(code_book.find(code)->second);
      // third
      code = 0xF & (tmp_short >> 8);
      w_float_tmp.push_back(code_book.find(code)->second);
    }
    // delete the last few redundant values
    for (int i = 0; i< (3 - short_len %3)%3; ++i){
      weights[layer_id - 1].pop_back();
    }
    weights.push_back(w_float_tmp);

  } else { // normal layer or squeeze1x1 layer
    // read the short
    int short_len = get_wb_stored_num(count_w[layer_id - 1], OTHER);
    cout << "-- normal short_len = " << short_len << endl;
    // if (fabs(short_len) > 600000) exit(-1);
    unsigned short *tmp_short_buf =
        (unsigned short *)calloc(short_len, sizeof(unsigned short));
    // tmp_short_ptr.reset(new unsigned short(short_len));
    in.read(reinterpret_cast<char *>(tmp_short_buf), short_len* sizeof(unsigned short)/ sizeof(char));

    std::vector<unsigned short> w_tmp;
    for (int i = 0; i < short_len; ++i) {
      w_tmp.push_back(tmp_short_buf[i]);
    }
    free(tmp_short_buf);
    weights_s.push_back(w_tmp);
 
    cout << "-- short formated weights or bias: " << endl;
    weight_check(weights_s, layer_id);
    // store the real values to weights
    std::vector<float> w_float_tmp;
    for (int i = 0; i < short_len; ++i) {
      unsigned short tmp_short = weights_s[layer_id - 1][i];
      // 1st
      unsigned char code = 0xF & tmp_short;
      w_float_tmp.push_back(code_book.find(code)->second);
      // 2nd
      code = 0xF & (tmp_short >> 4);
      w_float_tmp.push_back(code_book.find(code)->second);
      // 3rd
      code = 0xF & (tmp_short >> 8);
      w_float_tmp.push_back(code_book.find(code)->second);
      // 4th
      code = 0xF & (tmp_short >> 12);
      w_float_tmp.push_back(code_book.find(code)->second);
    }
    // delete the last few redundant values
    for (int i = 0; i< (4 - short_len %4)%4; ++i){
      weights[layer_id - 1].pop_back();
    }
    weights.push_back(w_float_tmp);
  }
}

int main(int argc, char **argv) {
  // check for the name of input binary files;
  if (argc < 2) {
    cout << "Error: no input file name provided!" << endl;
  }

  char *filename = argv[1];
  //char *filename = "inq_comp.binary";
  vector<char> min_exp_w;
  vector<char> min_exp_b;
  vector<int> count_w;
  vector<int> count_b;
  vector<vector<unsigned short> > weights_s;
  vector<vector<unsigned short> > bias_s;
  vector<vector<float> > weights;
  vector<vector<float> > bias;
/*
  char *buffer;
  if ((buffer = getcwd(NULL, 0)) == NULL) {
    perror("getcwd error");
  } else {
    printf("%s\n", buffer);
    free(buffer);
  }
*/
  ifstream in(filename, ios::in | ios::binary);
  if (!in.is_open()) {
    cout << "Error while opening file!" << endl;
    exit(-1);
  }
  // read data
  int layer_id = 0;

  while (!in.eof() && layer_id < 26) {
    layer_id++;
    cout << "=========================================================" << endl;
    cout << "layer_id = " << layer_id << endl;
    cout << "weights:" << endl;
    // read the weights
    read_half_layer(in, layer_id, min_exp_w, count_w, weights_s, weights, WEIGHT);
    cout << "----------------------------" << endl;
    // read the bias
    cout << "bias: " << endl;
    read_half_layer(in, layer_id, min_exp_b, count_b, bias_s, bias, BIAS);
    cout << endl;
  }

  // close the file
  in.close();
  cout << endl << endl;
/*
  // output some data for validation
  for (int i = 0; i < count_w.size(); ++i) {
    std::cout << "layer " << i + 1 << " weight count: " << count_w[i] << endl;
  }
*/
  return 0;
}

