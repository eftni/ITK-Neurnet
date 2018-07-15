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
        Dataset();
        Dataset(const std::string fname_im, const std::string fname_lab);
        /** Default destructor */
        virtual ~Dataset();

        /**
        * Load one image from dataset into curr_im
        */
        void load_one();

        /**
        * Returns the size of the current image
        * @return A pair containing the lenght and width
        */
        std::pair<uint32_t, uint32_t> get_size();

        /**
        * Returns the image currently stored
        * @return A array of 8bit integers
        */
        std::vector<std::vector<uint8_t>> get_im(){return curr_im;}

        /**
        * Returns what number the current image represents
        * @return Label of current image
        */
        uint8_t get_label(){return curr_label;}

        /**
        * Checks whether the end of the file has been reached
        * @return True if eof has not been reached, false otherwise
        */
        bool check_over();

        /**
        * Returns which image is currently loaded
        * @return Index of the current image
        */
        size_t get_index(){return index;}


    protected:

    private:
        std::ifstream fin_im; //!< Image filestream
        std::ifstream fin_lab; //!< Label filestream
        uint32_t magic_im; //!< Unique identifier of the image file
        uint32_t magic_lab; //!< Unique identifier of the label file
        uint32_t num_im; //!< Number of images in file
        uint32_t num_lab; //!< Number of labels in file
        uint32_t sizex; //!< Size of images along the x axis
        uint32_t sizey; //!< Size of images along the y axis
        size_t index; //!< Number of current image
        std::vector<std::vector<uint8_t>> curr_im; //!< Current image in matrix form
        uint8_t curr_label; //!< Current label
};

#endif // DATASET_H
