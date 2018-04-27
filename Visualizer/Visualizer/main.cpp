#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <utility>
#include <math.h>
#include <sstream>
#include "graphics.hpp"
#define XX 1400
#define YY 700

using namespace genv;

typedef std::vector<std::vector<std::vector<double>>> w_mat;

std::vector<double> separate(std::string s){
    std::stringstream ss;
    ss << s;
    std::vector<double> temp;
    double d;
    while(ss >> d){
        temp.push_back(d);
    }
    return temp;
}

w_mat load(std::string fname){
    w_mat w;
    std::vector<std::vector<double>> layer;
    std::ifstream fin(fname);
    if(!fin.good()){
        std::cout << "ERROR: FILE NOT FOUND" << std::endl;
        exit(1);
    }else{
        while(!fin.eof()){
            std::string temp;
            std::getline(fin, temp);
            if(temp.size() != 1){
                layer.push_back(separate(temp));
            }else{
                w.push_back(layer);
                layer.clear();
            }
        }
        return w;
    }
}

/*std::ostream& operator<<(std::ostream& out, w_mat w){
    for(std::vector<std::vector<double>> vv : w){
        for(std::vector<double> v : vv){
            for(double d : v){
                out << d << ' ';
            }
            out << std::endl;
        }
        out << "-------" << std::endl;
    }
    return out;
}*/

/*canvas gen_actmap(w_mat w, std::pair<int,int> neuron){
    canvas c(XX/2, XX/2);
    if(neuron.first == 1){

    }
    return c;
}*/ ///COMING IN FULL VERSION

/*template<typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T> v){
    for(T t : v){
        out << t << ' ';
    }
    out << std::endl;
    return out;
}*/

std::vector<std::vector<double>> row_to_mat(std::vector<double> row){
    std::vector<std::vector<double>> mat(sqrt(row.size()), std::vector<double>(sqrt(row.size()), 0));
    for(int i = 0; i < row.size(); ++i){
        int x = i/sqrt(row.size());
        int y = i - (x*sqrt(row.size()));
        mat[x][y] = row[i];
    }
    //std::cout << mat;
    return mat;
}

/*canvas gen_actmap(w_mat w, int neuron){
    std::vector<double> row(784, 0);
    double pos_max = 0, neg_min = 0;
    for(int i = 0; i < w[0].size(); ++i){
        row[i] = w[0][i][neuron];
        if(row[i] > pos_max) pos_max = row[i];
        if(row[i] < neg_min) neg_min = row[i];
    }
    std::vector<std::vector<double>> mat = row_to_mat(row);    /// MIGHT BE FLIPPED
    canvas c(4*YY/5, 4*YY/5);
    int boxsize = (4*YY/5)/mat.size();
    for(int i = 0; i < mat.size(); ++i){
        for(int j = 0; j < mat[i].size(); ++j){
            if(mat[i][j] >= 0){
                c << color(0,0,255*(mat[i][j]/pos_max));
            }else{
                c << color(255*(mat[i][j]/neg_min),0,0);
            }
            c << move_to(i*boxsize, j*boxsize) << box(boxsize, boxsize);
        }
    }
    return c;
}*/

std::vector<std::vector<double>> gen_actmap(w_mat w, std::pair<int, int> neuron){
    if(neuron.first == 0){
        std::vector<double> row(784, 0);
        double pos_max = 0, neg_min = 0;
        for(int i = 0; i < w[0].size(); ++i){
            row[i] = w[0][i][neuron.second];
            if(row[i] > pos_max) pos_max = row[i];
            if(row[i] < neg_min) neg_min = row[i];
        }
        std::vector<std::vector<double>> mat = row_to_mat(row);    /// MIGHT BE FLIPPED
        return mat;
    }else{
        std::vector<std::vector<double>> mat(28, std::vector<double>(28, 0));
        for(int i = 0; i < w[neuron.first].size(); ++i){
            mat += gen_actmap(w, std::pair<int, int> = {neuron.first-1, i}) * w[][][];  /// Write operators, figure out indexing
        }
        return mat;
    }
}

canvas draw_actmap(std::vector<std::vector<double>> mat){
    double pos_max = 0, neg_min  = 0;
    for(int i = 0; i < mat.size(); ++i){
        for(int j = 0; j < mat[i].size(); ++j){
            if(mat[i][j] > pos_max) pos_max = mat[i][j];
            if(mat[i][j] < neg_min) neg_min = mat[i][j];
        }
    }
    canvas c(4*YY/5, 4*YY/5);
    int boxsize = (4*YY/5)/mat.size();
    for(int i = 0; i < mat.size(); ++i){
        for(int j = 0; j < mat[i].size(); ++j){
            if(mat[i][j] >= 0){
                c << color(0,0,255*(mat[i][j]/pos_max));
            }else{
                c << color(255*(mat[i][j]/neg_min),0,0);
            }
            c << move_to(i*boxsize, j*boxsize) << box(boxsize, boxsize);
        }
    }
    return c;
}

canvas gen_net(w_mat w, std::pair<int,int> neuron = {0,0}){
    canvas c(XX/2, 4*YY/5);
    std::vector<std::vector<std::pair<int,int>>> coordinates;
    int xdiv = (XX/2)/(w.size()+1);
    for(int i = 0; i < w.size(); ++i){
        std::vector<std::pair<int,int>> temp;
        int ydiv = (4*YY/5)/(w[i][0].size()+1);
        for(int k = 0; k < w[i][0].size(); ++k){
            temp.push_back(std::make_pair((i+1)*xdiv, (k+1)*ydiv));
        }
        coordinates.push_back(temp);
    }
    //c << move_to(0,0) << color(255,0,0) << box(4*YY/5, 4*YY/5);
    for(std::vector<std::pair<int,int>> v : coordinates){
        for(std::pair<int,int> p : v){
            c << move_to(p.first-10, p.second-10) << color(255,255,255) << box(20,20) << move_to(p.first-5, p.second-5) << color(0,0,0) << box(10,10);
        }
    }
    /*if(neuron != std::make_pair(-1,-1)){
        c << move_to(coordinates[neuron.first][neuron.second].first-5, coordinates[neuron.first][neuron.second].second-5) << color(255,255,255) << box(10,10);
    }*/
    double pos_max = 0, neg_min = 0;
    for(int i = 1; i < w.size(); ++i){
        for(int j = 0; j < w[i].size(); ++j){
            for(int k = 0; k < w[i][j].size(); ++k){
                if(w[i][j][k] > pos_max) pos_max = w[i][j][k];
                if(w[i][j][k] < neg_min) neg_min = w[i][j][k];
            }
        }
    }
    for(int i = 1; i < w.size(); ++i){
        for(int j = 0; j < w[i].size(); ++j){
            for(int k = 0; k < w[i][j].size(); ++k){
                if(w[i][j][k] >= 0){
                    c << color(0,0,255*w[i][j][k]/pos_max);
                }else{
                    c << color(255*w[i][j][k]/neg_min,0,0);
                }
                c << move_to(coordinates[i-1][j].first+10, coordinates[i-1][j].second) << line_to(coordinates[i][k].first-10, coordinates[i][k].second);
            }
        }
    }
    return c;
}

void draw(std::string fname, std::pair<int, int> neuron){
    w_mat w = load(fname);
    canvas actmap = draw_actmap(gen_actmap(w, neuron));
    canvas net = gen_net(w, neuron);
    gout << move_to(YY/10-10, YY/10-10) << box_to(9*YY/10+10, 9*YY/10+10);
    gout << stamp(actmap, YY/10, YY/10) << stamp(net, 5*XX/10, YY/10) << refresh;
}

int main()
{
    w_mat w = load("..\\..\\MTLPERV2\\MTLPERV2\\w_mat0.txt");
    //w_mat w = load("..\\..\\MTLPERV1\\w_mat0.txt");
    canvas actmap = gen_actmap(w, 0);
    canvas net = gen_net(w);
    gout.open(XX,YY);
    gout << move_to(YY/10-10, YY/10-10) << box_to(9*YY/10+10, 9*YY/10+10);
    gout << stamp(actmap, YY/10, YY/10) << stamp(net, 5*XX/10, YY/10) << refresh;
    event ev;
    gin.timer(10);
    int index = 0;
    bool advance = 0;
    std::pair<int, int> neuron = {0, 0};
    while(gin >> ev && ev.keycode != 'q'){
        if(ev.keycode == key_space){
            advance = !advance;
        }
        if(ev.keycode == key_down && neuron != 15){
            std::stringstream ss;
            ss << index;
            std::string fname = "..\\..\\MTLPERV2\\MTLPERV2\\w_mat" + ss.str() + ".txt";
            //std::string fname = "..\\..\\MTLPERV1\\w_mat" + ss.str() + ".txt";
            ++neuron.second;
            draw(fname, neuron);
        }
        if(ev.keycode == key_up && neuron != 0){
            std::stringstream ss;
            ss << index;
            std::string fname = "..\\..\\MTLPERV2\\MTLPERV2\\w_mat" + ss.str() + ".txt";
            //std::string fname = "..\\..\\MTLPERV1\\w_mat" + ss.str() + ".txt";
            --neuron.second;
            draw(fname, neuron);
        }
        if(ev.type == ev_timer && advance){
            std::stringstream ss;
            ss << index;
            std::string fname = "..\\..\\MTLPERV2\\MTLPERV2\\w_mat" + ss.str() + ".txt";
            //std::string fname = "..\\..\\MTLPERV1\\w_mat" + ss.str() + ".txt";
            draw(fname, neuron);
            std::cout << index << std::endl;
            if(index%100 == 0){
                advance = 0;
            }
            if(index != 599){
                ++index;
            }else{
                //break;
                advance = 0;
            }
        }
    }
    return 0;
}
