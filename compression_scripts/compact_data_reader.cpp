
#include<iostream>
#include<vector>
#include<fstream>
#include<memory>
#include<map>
#include<cmath>

using namespace std;

enum Layer_type {SQUEEZE3x3, OTHER};

int get_wb_short_num(const int count, Layer_type lt) {
  if (lt == SQUEEZE3x3){
    return (count-1)/3 +1 ;
  } else{
    return (count-1)/2 +1 ;
  }
}

std::map<unsigned char, float> get_code_book(const char min, const int bits){
  std::map<unsigned char, float> code_book;

  for (unsigned char i = 0; i< (1<< (bits-1))-1; ++i){
    code_book.insert(pair<unsigned char, float>(i, -pow(2.0, min+i)));
  }
  code_book.insert(pair<unsigned char, float>(0, 0.0));

  unsigned char mid = 1<< (bits-1); // 8
  for (unsigned char i = mid; i< 1<< bits; ++i){
    code_book.insert(pair<unsigned char, float>(i, pow(2.0, min+i-mid)));
  }
  return code_book;
}

void read_half_layer(ifstream &in, int layer_id, vector<char> &min_exp_w, vector<int> &count_w, 
  vector<vector<unsigned short> > &weights_s, vector<vector<float> > &weights){

  int init_exp = -100;
  char tmp_c;
  int tmp_i;
  shared_ptr<unsigned short> tmp_cptr;
  // read the minimum exp for this block of data
  // min_exp_w.resize(layer_id, init_exp);
  in.read(&tmp_c, sizeof(char));
  min_exp_w.push_back(tmp_c);
  map<unsigned char, float> code_book;
  code_book = get_code_book(tmp_c, 4);
  // total number of data
  // count_w.resize(layer_id, 0);
  in.read(reinterpret_cast<char *>(&tmp_i), sizeof(int));
  count_w.push_back(tmp_i);
  weights_s.resize(layer_id);
  weights.resize(layer_id);

  // squeeze 3x3 layer
  if ((layer_id-1)%3 == 0 && layer_id != 1){
    // read the short 
    int short_len = get_wb_short_num(count_w[layer_id-1], SQUEEZE3x3);
    // weights_s[layer_id-1].resize(short_len, 0);
    tmp_cptr.reset(new unsigned short(short_len));
    in.read(reinterpret_cast<char *>(tmp_cptr.get()), short_len * sizeof(unsigned short));
    for (int i=0; i< short_len; ++i){
      weights_s[layer_id-1].push_back(tmp_cptr.get()[i]);
    }
    // store the real values to weights
    for (int i= 0; i< short_len; ++i){
      unsigned short tmp_short = weights_s[layer_id-1][i];
      // first
      unsigned char code = 0xF & tmp_short;
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // second
      code = 0xF & (tmp_short >> 4);
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // third
      code = 0xF & (tmp_short >> 8);
      weights[layer_id-1].push_back(code_book.find(code)->second);
    }
    // delete the last few redundant values
    if(short_len % 3 == 1){
      weights.pop_back();
      weights.pop_back();
    } else if (short_len % 3 == 2) {
      weights.pop_back();
    }
  } else { // normal layer or squeeze1x1 layer
    // read the short 
    int short_len = get_wb_short_num(count_w[layer_id-1], OTHER);
    // weights_s[layer_id-1].resize(short_len, 0);
    tmp_cptr.reset(new unsigned short(short_len));
    in.read(reinterpret_cast<char *>(tmp_cptr.get()), short_len * sizeof(unsigned short));
    for (int i=0; i< short_len; ++i){
      weights_s[layer_id-1].push_back(tmp_cptr.get()[i]);
    }
    // store the real values to weights
    for (int i= 0; i< short_len; ++i){
      unsigned short tmp_short = weights_s[layer_id-1][i];
      // first
      unsigned char code = 0xF & tmp_short;
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // second
      code = 0xF & (tmp_short >> 4);
      weights[layer_id-1].push_back(code_book.find(code)->second);
    }
    // delete the last few redundant values
    if(short_len % 2 == 1){
      weights.pop_back();
    }
  }
}


int main(int argc, char** argv){
  // check for the name of input binary files;
  if (argc < 2) {
    cout << "Error: no input file name provided!";
  }
  char *filename = argv[1];
  vector<char> min_exp_w;
  vector<char> min_exp_b;
  vector<int> count_w;
  vector<int> count_b;
  vector<vector<unsigned short> > weights_s;
  vector<vector<unsigned short> > bias_s;
  vector<vector<float> > weights;
  vector<vector<float> > bias;

  ifstream in (filename, ios::in | ios::binary );
  if (! in.is_open()){
    cout<< "Error while opening file!" << endl;
    exit(-1);
  }
  // read data
  int layer_id = 0;

  while (! in.eof()){
    layer_id++;
    // read the weights
    weights_s.resize(layer_id);
    weights.resize(layer_id);
    read_half_layer(in, layer_id, min_exp_w, count_w, weights_s, weights);
    // read the bias
    bias_s.resize(layer_id);
    bias.resize(layer_id);
    read_half_layer(in, layer_id, min_exp_b, count_b, bias_s, bias);
  }
  // close the file 
  in.close();
  // output some data for validation
  for (int i = 0; i< count_w.size(); ++i){
    std::cout << "layer " << i+1 << " weight count: " << count_w[i] << endl;
  }

  return 0;
}
