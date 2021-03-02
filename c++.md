#### override保留字

表示当前函数重写了基类的虚函数。 目的：

1.在函数比较多的情况下可以提示读者某个函数重写了基类虚函数（表示这个虚函数是从基类继承，不是派生类自己定义的）；
2.强制编译器检查某个函数是否重写基类虚函数，如果没有则报错。

#### 宏

宏也用三个点...来表示可变参数，`__VA_ARGS__`宏用来表示可变参数的内容，简单来说就是将左边宏中...的内容原样抄写在右边`__VA_ARGS__`所在的位置。如下例代码：

```
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

#define exampleNum(n) num##n
int num9=9;
int num=exampleNum(9); 将会扩展成 int num=num9;
```

#### 设计模式

 https://refactoringguru.cn/design-patterns/cpp

#### 初始化列表

 https://www.cnblogs.com/graphics/archive/2010/07/04/1770900.html

构造函数的执行可以分成两个阶段，初始化阶段和计算阶段(执行构造函数体内的赋值操作)，初始化阶段先于计算阶段。
类成员会在初始化阶段初始化，即使该成员没有出现在构造函数的初始化列表中。

必须使用初始化列表的情况：
1、常量成员，因为常量只能初始化不能赋值，所以必须放在初始化列表里面
2、引用类型，引用必须在定义的时候初始化，并且不能重新赋值，所以也要写在初始化列表里面
3、没有默认构造函数的类类型，因为使用初始化列表可以不必调用默认构造函数来初始化，而是直接调用拷贝构造函数初始化。

```
#include <iostream>
using namespace std;

struct Test1
{
    Test1() // 无参构造函数
    {
        cout << "Construct Test1" << endl ;
    }

    Test1(const Test1& t1) // 拷贝构造函数
    {
        cout << "Copy constructor for Test1" << endl ;
        this->a = t1.a ;
    }

    void operator=(const Test1& t1) // 赋值运算符
    {
        cout << "assignment for Test1" << endl ;
        this->a = t1.a ;
    }

    int a ;
};

//struct Test2
//{
//    Test1 test1 ;
//    Test2(const Test1 &t1)
//    {
//        test1 = t1 ;
//    }
//};

struct Test2
{
    Test1 test1 ;
    Test2(Test1 &t1):test1(t1) {}
};

int main()
{
    Test1 t1;
    Test2 t2(t1) ;
}
```

#### static_cast和dynamic_cast的区别

https://www.jianshu.com/p/5163a2678171
在类层次间进行上行转换时，dynamic_cast和static_cast的效果是一样的；在进行下行转换时，dynamic_cast具有类型检查的功能，比static_cast更安全。

```
class B
{
     virtual void f(){};
};
class D : public B
{
     virtual void f(){};
};
void main()
{
     B* pb = new D;   // unclear but ok
     B* pb2 = new B;
     D* pd = dynamic_cast<D*>(pb);   // ok: pb actually points to a D
     D* pd2 = dynamic_cast<D*>(pb2);   // pb2 points to a B not a D, now pd2 is NULL
}
```

#### 虚函数的实现机制

 https://www.cnblogs.com/malecrab/p/5572730.html

#### final关键字

用在类中表示该类不能被继承，用在方法中表示不能在子类中重写该方法。

#### 模板特化

#### extern

