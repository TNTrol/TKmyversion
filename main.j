.source Main.j
.class public Main
.super java/lang/Object

.method public static main([Ljava/lang/String;)V
.limit stack 2
.limit locals 2

invokestatic Main.main()V
return
.end method
.method public static test(III)V
.limit stack 3
.limit locals 4
iload 2
iconst_0
if_icmpgt label0
getstatic java/lang/System.out Ljava/io/PrintStream;
iload 0
invokevirtual java/io/PrintStream.println(I)V
return
label0:
iload 1
istore 3
iload 2
iconst_1
isub
istore 2
iload 0
iload 1
iadd
istore 1
iload 3
iload 1
iload 2
invokestatic Main.test(III)V
return
.end method
.method public static main()V
.limit stack 4
.limit locals 4
iconst_0
istore 0
iconst_1
istore 1
bipush 9
istore 3
iload 0
bipush 12
imul
i2f
ldc 7.9
fconst_1
fmul
fadd
f2i
istore 2
getstatic java/lang/System.out Ljava/io/PrintStream;
iload 2
invokevirtual java/io/PrintStream.println(I)V
iload 0
iload 1
iload 3
invokestatic Main.test(III)V
return
.end method
