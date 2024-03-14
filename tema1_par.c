// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT 16
#define FILENAME_MAX_SIZE 50
#define STEP 8
#define SIGMA 200
#define RESCALE_X 2048
#define RESCALE_Y 2048

#define CLAMP(v, min, max) \
    if (v < min)           \
    {                      \
        v = min;           \
    }                      \
    else if (v > max)      \
    {                      \
        v = max;           \
    }

typedef struct
{
    int thread_id;
    ppm_image *image;
    ppm_image **scaled_image;
    int step_x, step_y, P;
    pthread_barrier_t *barrier;
    ppm_image ***contour_map;
    unsigned char ***grid;

} ARGUMENTS;

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
void init_contour_map(ppm_image **map, int P, int thread_id)
{
    // calculate the iteration subdivisions
    int start = thread_id * (double)CONTOUR_CONFIG_COUNT / P;
    int end = (thread_id + 1) * (double)CONTOUR_CONFIG_COUNT / P;
    if (end > CONTOUR_CONFIG_COUNT)
    {
        end = CONTOUR_CONFIG_COUNT;
    }

    for (int i = start; i < end; i++)
    {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y)
{
    for (int i = 0; i < contour->x; i++)
    {
        for (int j = 0; j < contour->y; j++)
        {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
void sample_grid(ppm_image *image, unsigned char **grid, int step_x, int step_y, unsigned char sigma, int P, pthread_barrier_t *barrier, int thread_id)
{
    // calculate the iteration subdivisions
    int p = image->x / step_x;
    int q = image->y / step_y;

    int start = thread_id * (double)p / P;
    int end = (thread_id + 1) * (double)p / P;
    if (end > p) {
        end = p;
    }

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    for (int j = 0; j < q; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }

    // wait for threads to fincish to go the march function
    pthread_barrier_wait(barrier);
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y, int P, int thread_id)
{
    // calculate the iteration subdivisions
    int p = image->x / step_x;
    int q = image->y / step_y;

    int start = thread_id * (double)p / P;
    int end = (thread_id + 1) * (double)p / P;
    if (end > p) {
        end = p;
    }

    for (int i = start; i < end; i++)  {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *scaled_image, ppm_image **contour_map, unsigned char **grid, int p)
{
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= p; i++) {
        free(grid[i]);
    }
    free(grid);

    free(scaled_image->data);
    free(scaled_image);
}

void rescale_image(ppm_image *image, ppm_image **scaled_image, int P, pthread_barrier_t *barrier, int thread_id)
{
    uint8_t sample[3];
    // calculate the iteration subdivisions
    int start = thread_id * (double)(*scaled_image)->x / P;
    int end = (thread_id + 1) * (double)(*scaled_image)->x / P;
    if (end > (*scaled_image)->x) {
        end = (*scaled_image)->x;
    }

    // use bicubic interpolation for scaling

    for (int i = start; i < end; i++) {
        for (int j = 0; j < (*scaled_image)->y; j++) {

            float u = (float)i / (float)((*scaled_image)->x - 1);
            float v = (float)j / (float)((*scaled_image)->y - 1);
            sample_bicubic(image, u, v, sample);

            (*scaled_image)->data[i * (*scaled_image)->y + j].red = sample[0];
            (*scaled_image)->data[i * (*scaled_image)->y + j].green = sample[1];
            (*scaled_image)->data[i * (*scaled_image)->y + j].blue = sample[2];
        }
    }

    // wait for threads in order to go the first part of the marching squares algorithm
    pthread_barrier_wait(barrier);
}

void *thread_function(void *arg)
{
    ARGUMENTS *argument = (ARGUMENTS *)arg;
    int thread_id = argument->thread_id;
    int P = argument->P;
    ppm_image *image = argument->image;
    pthread_barrier_t *barrier = argument->barrier;
    ppm_image **scaled_image = argument->scaled_image;
    ppm_image **contour_map = *argument->contour_map;
    unsigned char **grid = *argument->grid;
    int step_x = argument->step_x;
    int step_y = argument->step_y;

    // 0. Initialize contour map
    init_contour_map(contour_map, P, thread_id);

    // if the image is too big
    if (image->x > RESCALE_X || image->y > RESCALE_Y)
    {
        // 1. Rescale the image
        rescale_image(image, scaled_image, P, barrier, thread_id);
    }

    // 2. Sample the grid
    sample_grid(*scaled_image, grid, step_x, step_y, SIGMA, P, barrier, thread_id);

    // 3. March the squares
    march(*scaled_image, grid, contour_map, step_x, step_y, P, thread_id);

    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    int P = atoi(argv[3]);
    pthread_t tid[P];
    int i;
    pthread_barrier_t barrier;
    ARGUMENTS arguments[P];

    // read image
    ppm_image *image = read_ppm(argv[1]);

    // initialize variables 
    int step_x = STEP;
    int step_y = STEP;
    int image_x = image->x;
    int image_y = image->y;

    // initialize and alloc memory for matrixes that store the results of the different functions
    // this is done in main because the threads would each alloc memory for them and we do not want that
    ppm_image **contour_map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!contour_map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    ppm_image *scaled_image;

    // we alloc memory for the scaled image only if the image is too big
    // otherwise scaled_image and image point to the same address and
    // it is a waste of space to alloc memory
    if (image_x > RESCALE_X || image_y > RESCALE_Y) {
        scaled_image = (ppm_image *)malloc(sizeof(ppm_image));
        if (!scaled_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
        scaled_image->x = RESCALE_X;
        scaled_image->y = RESCALE_Y;

        scaled_image->data = (ppm_pixel *)malloc(scaled_image->x * scaled_image->y * sizeof(ppm_pixel));
        if (!scaled_image)
        {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    } else {
        scaled_image = image;
    }

    // alloc memory depending on the scaled_image dimensions
    // initially, this part was made in the sample_grid function
    int p, q;
    if (image->x > RESCALE_X || image->y > RESCALE_Y) {
        p = RESCALE_X / step_x;
        q = RESCALE_X / step_y;
    } else {
        p = image->x / step_x;
        q = image->y / step_y;
    }

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++)
    {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i])
        {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    // initialize barrier
    pthread_barrier_init(&barrier, NULL, P);
    // the threads are being created
    for (i = 0; i < P; i++)
    {
        arguments[i].thread_id = i;
        arguments[i].contour_map = &contour_map;
        arguments[i].image = image;
        arguments[i].scaled_image = &scaled_image;
        arguments[i].grid = &grid;
        arguments[i].P = P;
        arguments[i].barrier = &barrier;
        arguments[i].step_x = step_x;
        arguments[i].step_y = step_y;
        pthread_create(&tid[i], NULL, thread_function, &arguments[i]);
    }

    // we wait for the threads to finish
    for (i = 0; i < P; i++)
    {
        pthread_join(tid[i], NULL);
    }
    pthread_barrier_destroy(&barrier);

    // 4. Write output
    write_ppm(scaled_image, argv[2]);

    // because in the case of the initial image being too big
    // we also dealloc the initial image and not only the  scaled one
    if (image_x > RESCALE_X || image_y > RESCALE_Y)
    {
        free(image->data);
        free(image);
    }
    free_resources(scaled_image, contour_map, grid, p);

    return 0;
}
