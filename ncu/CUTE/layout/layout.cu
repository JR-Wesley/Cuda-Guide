#include <cute/tensor.hpp>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}

int main() {
    Layout s8 = make_layout(Int<8>{});
    Layout d8 = make_layout(8);

    Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

    Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                            make_stride(Int<12>{},Int<1>{}));
    Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                            LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                            LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
// Layout s2xh4_col = make_layout(shape(s2xh4),
//                                LayoutLeft{});
    std::cout << "s8:           " << s8 << std::endl;
    std::cout << "d8:           " << d8 << std::endl;
    std::cout << "s2xs4:        " << s2xs4 << std::endl;
    std::cout << "s2xd4:        " << s2xd4 << std::endl;
    std::cout << "s2xd4_a:      " << s2xd4_a << std::endl;
    std::cout << "s2xd4_col:    " << s2xd4_col << std::endl;   
    std::cout << "s2xd4_row:    " << s2xd4_row << std::endl;
    std::cout << "s2xh4:        " << s2xh4 << std::endl;

    print2D(s2xd4);
    print2D(s2xd4_a);
    print2D(s2xh4);
    print2D(s2xh4);
    print_layout(s2xh4);
    return 0;
}