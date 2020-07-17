#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <omp.h>

/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int** inputImage;
int** outputImage;
int num_threads;
int maxChunk;
int chunkCounter;
std::mutex chunkLock;

/* ****************Change and add functions below ***************** */

int getChunk(int &start, int &end);

void prewitt(int threadNum);

void dispatch_threads();

/* ****************Need not to change the function below ***************** */

int main(int argc, char* argv[])
{
    if(argc != 5)
    {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename> <# of Threads> <# of Chunks>" << std::endl;
        return 0;
    }

    std::ifstream file(argv[1]);
    if(!file.is_open())
    {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    num_threads = std::atoi(argv[3]);
    /* maxChunk is total number of chunks to process */
    maxChunk = std::atoi(argv[4]);

    std::cout << "Detect edges in " << argv[1] << " using " << num_threads << " threads\n" << std::endl;

    /* ******Reading image into 2-D array below******** */

    std::string workString;
    /* Remove comments '#' and check image format */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            if( workString.at(1) != '2' ){
                std::cout << "Input image is not a valid PGM image" << std::endl;
                return 0;
            } else {
                break;
            }
        } else {
            continue;
        }
    }
    /* Check image size */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        } else {
            continue;
        }
    }

    inputImage = new int*[image_height];
    outputImage = new int*[image_height];
    for(int i = 0; i < image_height; ++i){
        inputImage[i] = new int[image_width];
        outputImage[i] = new int[image_width];
    }

    /* Check image max shades */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        } else {
            continue;
        }
    }
    /* Fill input image matrix */
    int pixel_val;
    for( int i = 0; i < image_height; i++ )
    {
        if( std::getline(file,workString) && workString.at(0) != '#' ){
            std::stringstream stream(workString);
            for( int j = 0; j < image_width; j++ ){
                if( !stream )
                    break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        } else {
            continue;
        }
    }

    /************ Function that creates threads and manage dynamic allocation of chunks *********/
    double dtime = omp_get_wtime();
    dispatch_threads();
    dtime = omp_get_wtime() - dtime;
    std::cout << "Took " << dtime << " second(s).\n" << std::endl;

    /* ********Start writing output to your file************ */
    std::ofstream ofile(argv[2]);
    if( ofile.is_open() )
    {
        ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
        for( int i = 0; i < image_height; i++ )
        {
            for( int j = 0; j < image_width; j++ ){
                ofile << outputImage[i][j] << " ";
            }
            ofile << "\n";
        }
    } else {
        std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
        return 0;
    }
    return 0;
}

void dispatch_threads()
{
    chunkCounter = 0;
    std::vector<std::thread> threads;
    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(std::thread(&prewitt, i));
    }
    for(int i = 0; i < num_threads; ++i)
    {
        threads[i].join();
    }
}

int getChunk(int &start, int &end)
{
    int numberOfRows = static_cast<int>(ceil(static_cast<double>(image_height)/maxChunk));
    start = chunkCounter * numberOfRows;
    end = start + numberOfRows;
    if(end > image_height)
        end = image_height;
    return chunkCounter+1;
}

void prewitt(int threadNum)
{
    int maskX[3][3] = {
        {1,0,-1},
        {1,0,-1},
        {1,0,-1}
    };
    int maskY[3][3] = {
        {1,1,1},
        {0,0,0},
        {-1,-1,-1}
    };
    int start, end;
    while(1) {
        chunkLock.lock();
        chunkCounter = getChunk(start, end);
        if(start >= image_height) {
//            std::cout << "Thread " << threadNum << " joining." << std::endl<< std::endl;
            chunkLock.unlock();
            break;
        }
//        std::cout << "Thread " << threadNum << " processing row: " << start << " and ending at: " << end << std::endl << std::endl;
        chunkLock.unlock();
        for(int x = start; x < end; ++x){
            for(int y = 0; y < image_width; ++y){
                int grad_x = 0, grad_y = 0, grad = 0;
                if(x == 0 || x ==(image_height-1) || y == 0 || y == (image_width-1))
                    grad = 0;
                else {
                    for(int i = -1; i<= 1; ++i) {
                        for(int j = -1; j <= 1; ++j) {
                            grad_x += (inputImage[x+i][y+j] * maskX[i+1][j+1]);
                            grad_y += (inputImage[x+i][y+j] * maskY[i+1][j+1]);
                        }
                    }
                    grad = static_cast<int>(sqrt((grad_x * grad_x) + (grad_y * grad_y)));
                }
//                std::cout << "Grad[" << x << "][" << y << "]: " << grad << std::endl;
                if(grad < 0)
                    grad = 0;
                else if(grad > 255)
                    grad = 255;
                outputImage[x][y] = grad;
            }
        }
    }
}
