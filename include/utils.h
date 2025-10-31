#ifdef UTILS_IMPLEMENTATION

#define CHECK_ERR(resp) do { \
    cudaError_t _err = (resp); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(_err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#endif
