# Layout

## 本地编译入口

原构建入口已迁移为 [commands/compile/run.sh](commands/compile/run.sh)，只编译、不运行。显式指定本机支持的 `CUDA_ARCH`；例如适用时执行 `CUDA_ARCH=sm_89 bash commands/compile/run.sh`。日志位于该动作的 `output.log`，共享产物位于 `build/<CUDA_ARCH>/<Release或Debug>/`。参数、依赖与仅静态验证的边界见 [NCU 导航](../../README.md)。本文源码讲解与既有记录未因入口迁移而重新实测。

首先，我们要知道，`Shape`等这些基本类型都是 `cute::tuple` 的别名。

## 源码解读

`/cutlass/include/cute/layout.hpp`定义了实现。

```cpp

// Aliases

template <class... Shapes>
using Shape = cute::tuple<Shapes...>;

template <class... Strides>
using Stride = cute::tuple<Strides...>;

template <class... Strides>
using Step = cute::tuple<Strides...>;

template <class... Coords>
using Coord = cute::tuple<Coords...>;

template <class... Layouts>
using Tile = cute::tuple<Layouts...>;

template <class... Ts>
CUTE_HOST_DEVICE constexpr
Shape<Ts...>
make_shape(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Stride<Ts...>
make_stride(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Step<Ts...>
make_step(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Coord<Ts...>
make_coord(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Tile<Ts...>
make_tile(Ts const&... t)
{
  return {t...};
}

```

### 1. 别名模板定义

```cpp
template <class... Shapes>
using Shape = std::tuple<Shapes...>;
```

- **语法解释**：
  - `template <class... Shapes>`：声明一个**模板参数包**，可以接受任意数量、任意类型的类型参数（如 `int`, `float`, `Int<4>`, `Goto<3>` 等）。
  - `using Shape = std::tuple<Shapes...>;`：定义一个**类型别名模板**，名为 `Shape`，它等价于一个 `std::tuple`，其元素类型就是传入的 `Shapes...`。

- **作用**：
  - `Shape` 是一个**元组类型**，用来表示**多维数组的形状（shape）**，比如 `(4, 8, 16)` 表示一个三维张量。
  - 例如：`Shape<int, int, Int<4>>` 表示一个包含 3 个维度的形状，前两个是运行时 `int`，第三个是编译时常量 `4`。
  - 使用 `using` 而不是 `typedef` 是因为 `using` 更清晰，支持模板别名。

---

### 2.  函数模板

```cpp
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Shape<Ts...>
make_shape(Ts const&... t) {
  return {t...};
}
```

`make_shape`, `make_stride`, `make_step`, `make_coord`, `make_tile` 都是工厂函数

- **语法解释**：
  - `template <class... Ts>`：模板参数包，接受任意数量的类型。
  - `CUTE_HOST_DEVICE`：这是一个宏，通常定义为 `__host__ __device__`，表示该函数可以在 **CPU (host)** 和 **GPU (device)** 上编译和运行。这是 CUDA 编程的关键。
  - `constexpr`：表示该函数可以在编译时求值，有助于性能优化。
  - `Shape<Ts...>`：返回类型是前面定义的 `Shape` 别名，即 `std::tuple<Ts...>`。
  - `Ts const&... t`：函数参数是一个**值参数包**，接受任意数量的常量引用。
  - `{t...}`：使用**初始化列表**和**参数包展开**，将所有参数 `t` 展开并初始化一个 `std::tuple`。

```cpp
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#  define CUTE_DEVICE      __forceinline__          __device__
#  define CUTE_HOST        __forceinline__ __host__
```

- 以`make_shape`为例：
  - 这是一个**工厂函数**，用于创建一个 `Shape` 类型的对象（即 `std::tuple`）。
  - 它将传入的多个参数打包成一个元组，表示张量的维度大小。
  - **示例**：
    ```cpp
    auto s = make_shape(4, 8, 16);  // 返回 Shape<int,int,int> 即 tuple(4,8,16)
    ```

---

### 详细拆解

非常好的问题！我们来深入、详细地拆解您提到的 `make_shape` 工厂函数，以及 `auto shape = make_shape(4, 8, 16);` 这样的语句在 C++ 编译器中是如何一步步展开和工作的。

---

**工厂函数（Factory Function）** 是一种设计模式，它的作用是**封装对象的创建过程**，而不是让用户直接使用构造函数或复杂的类型声明。

```cpp
auto shape = make_shape(4, 8, 16);
```

- 不用写 `std::tuple<int,int,int>{4,8,16}` 这种冗长的类型。
- 也不用写 `Shape<int,int,int>{4,8,16}`。
- 给出“一个 shape，参数是 4,8,16”，函数就自动推导类型并创建对象。

✅ **这就是工厂函数的核心价值：简化创建、类型推导、语义清晰。**

---
第 1 步：调用 `make_shape(4, 8, 16)`

```cpp
auto shape = make_shape(4, 8, 16);
```

你传入了三个 `int` 类型的字面量：`4`, `8`, `16`。

第 2 步：模板参数推导（Template Argument Deduction）
函数模板是：
```cpp
template <class... Ts>
Shape<Ts...> make_shape(Ts const&... t)
```

编译器看到你传了 `(4, 8, 16)`，都是 `int` 类型，于是：
- 推导出 `Ts...` 是 `int, int, int`
- 所以 `Ts const&... t` 变成 `(int const&, int const&, int const&)`

✅ **模板参数包 `Ts...` 被展开为 `int, int, int`**

第 3 步：确定返回类型

返回类型是：

```cpp
Shape<Ts...> → Shape<int, int, int>
```

而 `Shape` 是一个别名模板：

```cpp
template <class... Shapes>
using Shape = std::tuple<Shapes...>;
```

所以：

```cpp
Shape<int, int, int> → std::tuple<int, int, int>
```

✅ **最终返回类型是 `std::tuple<int, int, int>`**

第 4 步：函数体执行 `return {t...};`

函数体是：

```cpp
return {t...};
```

- `t...` 是参数包，包含 `4, 8, 16`
- `{t...}` 展开为 `{4, 8, 16}`
- 这是一个**初始化列表**，用来构造一个 `std::tuple<int, int, int>`

等价于：

```cpp
return std::tuple<int, int, int>{4, 8, 16};
```

第 5 步：`auto` 类型推导

```cpp
auto shape = make_shape(4, 8, 16);
```

- `auto` 会自动推导右边表达式的类型
- `make_shape(4,8,16)` 返回 `std::tuple<int,int,int>`
- 所以 `shape` 的类型就是 `std::tuple<int,int,int>`

✅ **最终：`shape` 是一个包含三个整数的元组，值为 (4,8,16)**

---

总结

| 语法 | 展开过程 | 最终结果 |
|------|----------|---------|
| `make_shape(4, 8, 16)` | `Ts...` → `int,int,int` | `Shape<int,int,int>` → `tuple<int,int,int>` |
| `return {t...}` | `{t...}` → `{4,8,16}` | `return tuple{4,8,16};` |
| `auto shape = ...` | `auto` 推导返回类型 | `shape` 是 `tuple<int,int,int>` |
| `make_shape(1, 2)` | `Ts...` → `int,int` | `tuple<int,int>{1,2}` |

---
