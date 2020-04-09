
#include <iostream>

#if defined(__AVX512F__)
#pragma message ("__AVX512F__ defined")
#endif

#if defined(__AVX__)
#pragma message ("__AVX__ defined")
#endif

#if defined(__SSE__)
#pragma message ("__SSE__ defined")
#endif

#if defined(__SSE4_1__)
#pragma message ("__SSE4_1__ defined")
#endif

#if defined(USE_SIMD)
#pragma message ("USE_SIMD defined")
#endif

#if defined(__SSE2__)
#pragma message ("__SSE2__ defined")
#endif

#if defined(__AVX512ER__)
#pragma message ("__AVX512ER__ defined")
#endif

#if defined(__SSE3__)
#pragma message ("__SSE3__ defined")
#endif

#if defined(USE_CL)
#pragma message ("USE_CL defined")
#endif

#if defined(USE_SIMD_EWALD)
#pragma message ("USE_SIMD_EWALD defined")
#endif

int main() {

	std::cout << "Hello world!" << std::endl;
	return 0;

}
