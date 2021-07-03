@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00"

@formatInt = private constant [3 x i8] c"%d\0A"
@formatFloat = private constant [3 x i8] c"%f\0A"
@formatChar = private constant [3 x i8] c"%c\0A"


declare i32 @printf(i8*, ...) nounwind
; External declaration of the puts function

; Definition of main function
define i32 @main() {   ; i32()*
entry:
  ; Convert [13 x i8]* to i8*...
  ;%x = shl i32 2, 3
;   %q = alloca float ;ptr
;   store float 3.1, float* %q ;save to mem
;   %q = load float, float* %q
  
  %x = fadd double 1.3, 1.3
  %c = add i8 96, 1
    
  ; Call puts function to write out the string to stdout.
  ;%call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @formatInt, i32 0, i32 0), i32 %x) ;int
  ;%call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @formatFloat, i32 0, i32 0), double %x) ;float
  ;%call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @formatFloat, i32 0, i32 0), i8 %c) ;char
  ret i32 0
}

