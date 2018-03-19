#ifndef DATASET_H
#define DATASET_H
#include "vector"
#include "utility"
#include "stdint.h"
#include "fstream"
#include "stdlib.h"
#include "iostream"

class Dataset
{
    public:
        /** Default constructor */
        Dataset(std::string fname_im, std::string fname_lab);
        /** Default destructor */
        virtual ~Dataset();
        void load_one();
        std::pair<uint32_t, uint32_t> get_size();
        std::vector<std::vector<uint8_t>> get_im(){return curr_im;}
        uint8_t get_label(){return curr_label;}


    protected:

    private:
        uint32_t magic_im; //!< Unique identifier of the image file
        uint32_t magic_lab; //!< Unique identifier of the label file
        uint32_t num_im; //!< Number of images in file
        uint32_t num_lab; //!< Number of labels in file
        uint32_t sizex; //!< Size of images along the x axis
        uint32_t sizey; //!< Size of images along the y axis
        int index; //!< Number of current image
        std::ifstream fin_im; //!< Image filestream
        std::ifstream fin_lab; //!< Label filestream
        std::vector<std::vector<uint8_t>> curr_im; //!< Current image in matrix form
        uint8_t curr_label; //!< Current label
};

#endif // DATASET_H
