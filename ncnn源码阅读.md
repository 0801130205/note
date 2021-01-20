Extractor有个私有对象ExtractorPrivate

```
//class ExtractorPrivate
public:
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;
```

Net里有个私有对象NetPrivate

```
//class NetPrivate
public:
	//不存放具体的计算数据，真正的blob数据存放在blob_mats里
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;
    Option& opt;
```

```
//class Blob
// blob name
std::string name;
// layer index which produce this blob as output
int producer;
// layer index which need this blob as input
int consumer;
// shape hint
Mat shape;
```

## 网络和运算是分开的

```
ncnn的net是网络模型，实际使用的是extractor，
也就是同个net可以有很多个运算实例，而且运算实例互不影响，中间结果保留在extractor内部，
在多线程使用时共用网络的结构和参数数据，初始化网络模型和参数只需要一遍.

举个例子：全局静态的net实例，初始化一次后，就能不停地生成extractor使用.
```

```
[layer type] [layer name] [input count] [output count] [input blobs] [output blobs] [layer specific params]
```



```
宏也用三个点...来表示可变参数，__VA_ARGS__宏用来表示可变参数的内容，简单来说就是将左边宏中...的内容原样抄写在右边__VA_ARGS__所在的位置。如下例代码：

#include <stdio.h>
#define debug(...) printf(__VA_ARGS__)
int main(void)
{
    int year = 2018；
    debug("this year is %d\n", year);  //效果同printf("this year is %d\n", year);
}
```



```
#define example(instr) printf("the input string is:/t%s/n",#instr)
example(abc)； //在编译时将会展开成：printf("the input string is:/t%s/n","abc");
```

