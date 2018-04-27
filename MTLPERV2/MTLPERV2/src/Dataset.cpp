#include "Dataset.h"

Dataset::Dataset(){

}

Dataset::Dataset(const std::string fname_im, const std::string fname_lab) :
fin_im(fname_im, std::ifstream::binary),
fin_lab(fname_lab, std::ifstream::binary),
index(0)
{
    if(!fin_im.good() || !fin_lab.good()){
        std::cout << "Error: Can't find files" << std::endl;
        exit(1);
    }
    uint8_t b[4];
    fin_im.read(reinterpret_cast<char *>(&b), sizeof(b));
    magic_im = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    fin_im.read(reinterpret_cast<char *>(&b), sizeof(b));
    num_im = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    fin_im.read(reinterpret_cast<char *>(&b), sizeof(b));
    sizex = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    fin_im.read(reinterpret_cast<char *>(&b), sizeof(b));
    sizey = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    fin_lab.read(reinterpret_cast<char *>(&b), sizeof(b));
    magic_lab = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    fin_lab.read(reinterpret_cast<char *>(&b), sizeof(b));
    num_lab = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
    if(magic_lab != 2049 || magic_im != 2051){
        std::cout << "Error: Incorrect magic numbers. Check file integrity." << std::endl;
        exit(1);
    }
    if(num_im != num_lab){
        std::cout << "Error: File sizes don't match" << std::endl;
        exit(1);
    }
    load_one();
    index = 1;
}

Dataset::~Dataset()
{
    //dtor
}

void Dataset::load_one(){
    ++index;
    std::vector<std::vector<uint8_t>> temp(sizey, std::vector<uint8_t>(sizex, 0));
    for(size_t y = 0; y < sizey; ++y){
        for(size_t x = 0; x < sizex; ++x){
            uint8_t c = 0;
            fin_im.read(reinterpret_cast<char *>(&c), sizeof(c));
            temp[x][y] = c;
        }
    }
    curr_im = temp;
    uint8_t c = 0;
    fin_lab.read(reinterpret_cast<char *>(&c), sizeof(c));
    curr_label = c;
}

std::pair<uint32_t, uint32_t> Dataset::get_size(){
    return std::make_pair(sizex, sizey);
}

bool Dataset::check_over(){
    if(index <= num_im){ ///CHECK
        return true;
    }else{
        return false;
    }
}
