ù
Ê
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8

hiddenLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*$
shared_namehiddenLayer1/kernel
}
'hiddenLayer1/kernel/Read/ReadVariableOpReadVariableOphiddenLayer1/kernel* 
_output_shapes
:
¬*
dtype0
{
hiddenLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*"
shared_namehiddenLayer1/bias
t
%hiddenLayer1/bias/Read/ReadVariableOpReadVariableOphiddenLayer1/bias*
_output_shapes	
:¬*
dtype0

hiddenLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬d*$
shared_namehiddenLayer2/kernel
|
'hiddenLayer2/kernel/Read/ReadVariableOpReadVariableOphiddenLayer2/kernel*
_output_shapes
:	¬d*
dtype0
z
hiddenLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namehiddenLayer2/bias
s
%hiddenLayer2/bias/Read/ReadVariableOpReadVariableOphiddenLayer2/bias*
_output_shapes
:d*
dtype0

outputLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*#
shared_nameoutputLayer/kernel
y
&outputLayer/kernel/Read/ReadVariableOpReadVariableOpoutputLayer/kernel*
_output_shapes

:d
*
dtype0
x
outputLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameoutputLayer/bias
q
$outputLayer/bias/Read/ReadVariableOpReadVariableOpoutputLayer/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
­
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*è
valueÞBÛ BÔ
ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
6
!iter
	"decay
#learning_rate
$momentum
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
­
regularization_losses
trainable_variables

%layers
&metrics
	variables
'non_trainable_variables
(layer_metrics
)layer_regularization_losses
 
 
 
 
­
regularization_losses
trainable_variables

*layers
+metrics
	variables
,non_trainable_variables
-layer_metrics
.layer_regularization_losses
_]
VARIABLE_VALUEhiddenLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
trainable_variables

/layers
0metrics
	variables
1non_trainable_variables
2layer_metrics
3layer_regularization_losses
_]
VARIABLE_VALUEhiddenLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
trainable_variables

4layers
5metrics
	variables
6non_trainable_variables
7layer_metrics
8layer_regularization_losses
^\
VARIABLE_VALUEoutputLayer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEoutputLayer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
trainable_variables

9layers
:metrics
	variables
;non_trainable_variables
<layer_metrics
=layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

>0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	@total
	Acount
B	variables
C	keras_api
D
	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

B	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

G	variables

 serving_default_inputLayer_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
¼
StatefulPartitionedCallStatefulPartitionedCall serving_default_inputLayer_inputhiddenLayer1/kernelhiddenLayer1/biashiddenLayer2/kernelhiddenLayer2/biasoutputLayer/kerneloutputLayer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_39755
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'hiddenLayer1/kernel/Read/ReadVariableOp%hiddenLayer1/bias/Read/ReadVariableOp'hiddenLayer2/kernel/Read/ReadVariableOp%hiddenLayer2/bias/Read/ReadVariableOp&outputLayer/kernel/Read/ReadVariableOp$outputLayer/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_39979
ï
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehiddenLayer1/kernelhiddenLayer1/biashiddenLayer2/kernelhiddenLayer2/biasoutputLayer/kerneloutputLayer/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_40031Î

÷
F__inference_outputLayer_layer_call_and_return_conditional_losses_39563

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

û
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39529

inputs2
matmul_readvariableop_resource:
¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ù
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39546

inputs1
matmul_readvariableop_resource:	¬d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
'
é
__inference__traced_save_39979
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¦
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*[
_input_shapesJ
H: :
¬:¬:	¬d:d:d
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
¬:!

_output_shapes	
:¬:%!

_output_shapes
:	¬d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

û
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39865

inputs2
matmul_readvariableop_resource:
¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
²
E__inference_sequential_layer_call_and_return_conditional_losses_39660

inputs&
hiddenlayer1_39644:
¬!
hiddenlayer1_39646:	¬%
hiddenlayer2_39649:	¬d 
hiddenlayer2_39651:d#
outputlayer_39654:d

outputlayer_39656:

identity¢$hiddenLayer1/StatefulPartitionedCall¢$hiddenLayer2/StatefulPartitionedCall¢#outputLayer/StatefulPartitionedCallÚ
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_395162
inputLayer/PartitionedCallÃ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39644hiddenlayer1_39646*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_395292&
$hiddenLayer1/StatefulPartitionedCallÌ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_39649hiddenlayer2_39651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395462&
$hiddenLayer2/StatefulPartitionedCallÇ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39654outputlayer_39656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_395632%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÂ
NoOpNoOp%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

÷
F__inference_outputLayer_layer_call_and_return_conditional_losses_39905

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ß
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_39849

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
E__inference_sequential_layer_call_and_return_conditional_losses_39712
inputlayer_input&
hiddenlayer1_39696:
¬!
hiddenlayer1_39698:	¬%
hiddenlayer2_39701:	¬d 
hiddenlayer2_39703:d#
outputlayer_39706:d

outputlayer_39708:

identity¢$hiddenLayer1/StatefulPartitionedCall¢$hiddenLayer2/StatefulPartitionedCall¢#outputLayer/StatefulPartitionedCallä
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_395162
inputLayer/PartitionedCallÃ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39696hiddenlayer1_39698*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_395292&
$hiddenLayer1/StatefulPartitionedCallÌ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_39701hiddenlayer2_39703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395462&
$hiddenLayer2/StatefulPartitionedCallÇ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39706outputlayer_39708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_395632%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÂ
NoOpNoOp%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
§	

*__inference_sequential_layer_call_fn_39826

inputs
unknown:
¬
	unknown_0:	¬
	unknown_1:	¬d
	unknown_2:d
	unknown_3:d

	unknown_4:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_395702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

,__inference_hiddenLayer2_layer_call_fn_39894

inputs
unknown:	¬d
	unknown_0:d
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs

¼
E__inference_sequential_layer_call_and_return_conditional_losses_39732
inputlayer_input&
hiddenlayer1_39716:
¬!
hiddenlayer1_39718:	¬%
hiddenlayer2_39721:	¬d 
hiddenlayer2_39723:d#
outputlayer_39726:d

outputlayer_39728:

identity¢$hiddenLayer1/StatefulPartitionedCall¢$hiddenLayer2/StatefulPartitionedCall¢#outputLayer/StatefulPartitionedCallä
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_395162
inputLayer/PartitionedCallÃ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39716hiddenlayer1_39718*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_395292&
$hiddenLayer1/StatefulPartitionedCallÌ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_39721hiddenlayer2_39723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395462&
$hiddenLayer2/StatefulPartitionedCallÇ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39726outputlayer_39728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_395632%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÂ
NoOpNoOp%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
þ

,__inference_hiddenLayer1_layer_call_fn_39874

inputs
unknown:
¬
	unknown_0:	¬
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_395292
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ=
ü
!__inference__traced_restore_40031
file_prefix8
$assignvariableop_hiddenlayer1_kernel:
¬3
$assignvariableop_1_hiddenlayer1_bias:	¬9
&assignvariableop_2_hiddenlayer2_kernel:	¬d2
$assignvariableop_3_hiddenlayer2_bias:d7
%assignvariableop_4_outputlayer_kernel:d
1
#assignvariableop_5_outputlayer_bias:
%
assignvariableop_6_sgd_iter:	 &
assignvariableop_7_sgd_decay: .
$assignvariableop_8_sgd_learning_rate: )
assignvariableop_9_sgd_momentum: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity£
AssignVariableOpAssignVariableOp$assignvariableop_hiddenlayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hiddenlayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_hiddenlayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_outputlayer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_outputlayer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6 
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¡
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14f
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_15ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Å	

*__inference_sequential_layer_call_fn_39585
inputlayer_input
unknown:
¬
	unknown_0:	¬
	unknown_1:	¬d
	unknown_2:d
	unknown_3:d

	unknown_4:

identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_395702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
þ
²
E__inference_sequential_layer_call_and_return_conditional_losses_39570

inputs&
hiddenlayer1_39530:
¬!
hiddenlayer1_39532:	¬%
hiddenlayer2_39547:	¬d 
hiddenlayer2_39549:d#
outputlayer_39564:d

outputlayer_39566:

identity¢$hiddenLayer1/StatefulPartitionedCall¢$hiddenLayer2/StatefulPartitionedCall¢#outputLayer/StatefulPartitionedCallÚ
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_395162
inputLayer/PartitionedCallÃ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39530hiddenlayer1_39532*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_395292&
$hiddenLayer1/StatefulPartitionedCallÌ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_39547hiddenlayer2_39549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395462&
$hiddenLayer2/StatefulPartitionedCallÇ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39564outputlayer_39566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_395632%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÂ
NoOpNoOp%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

#__inference_signature_wrapper_39755
inputlayer_input
unknown:
¬
	unknown_0:	¬
	unknown_1:	¬d
	unknown_2:d
	unknown_3:d

	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_395032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
Å+

 __inference__wrapped_model_39503
inputlayer_inputJ
6sequential_hiddenlayer1_matmul_readvariableop_resource:
¬F
7sequential_hiddenlayer1_biasadd_readvariableop_resource:	¬I
6sequential_hiddenlayer2_matmul_readvariableop_resource:	¬dE
7sequential_hiddenlayer2_biasadd_readvariableop_resource:dG
5sequential_outputlayer_matmul_readvariableop_resource:d
D
6sequential_outputlayer_biasadd_readvariableop_resource:

identity¢.sequential/hiddenLayer1/BiasAdd/ReadVariableOp¢-sequential/hiddenLayer1/MatMul/ReadVariableOp¢.sequential/hiddenLayer2/BiasAdd/ReadVariableOp¢-sequential/hiddenLayer2/MatMul/ReadVariableOp¢-sequential/outputLayer/BiasAdd/ReadVariableOp¢,sequential/outputLayer/MatMul/ReadVariableOp
sequential/inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential/inputLayer/Const´
sequential/inputLayer/ReshapeReshapeinputlayer_input$sequential/inputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/inputLayer/Reshape×
-sequential/hiddenLayer1/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02/
-sequential/hiddenLayer1/MatMul/ReadVariableOpÜ
sequential/hiddenLayer1/MatMulMatMul&sequential/inputLayer/Reshape:output:05sequential/hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
sequential/hiddenLayer1/MatMulÕ
.sequential/hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype020
.sequential/hiddenLayer1/BiasAdd/ReadVariableOpâ
sequential/hiddenLayer1/BiasAddBiasAdd(sequential/hiddenLayer1/MatMul:product:06sequential/hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
sequential/hiddenLayer1/BiasAdd¡
sequential/hiddenLayer1/ReluRelu(sequential/hiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential/hiddenLayer1/ReluÖ
-sequential/hiddenLayer2/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02/
-sequential/hiddenLayer2/MatMul/ReadVariableOpß
sequential/hiddenLayer2/MatMulMatMul*sequential/hiddenLayer1/Relu:activations:05sequential/hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
sequential/hiddenLayer2/MatMulÔ
.sequential/hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential/hiddenLayer2/BiasAdd/ReadVariableOpá
sequential/hiddenLayer2/BiasAddBiasAdd(sequential/hiddenLayer2/MatMul:product:06sequential/hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
sequential/hiddenLayer2/BiasAdd 
sequential/hiddenLayer2/ReluRelu(sequential/hiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
sequential/hiddenLayer2/ReluÒ
,sequential/outputLayer/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02.
,sequential/outputLayer/MatMul/ReadVariableOpÜ
sequential/outputLayer/MatMulMatMul*sequential/hiddenLayer2/Relu:activations:04sequential/outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential/outputLayer/MatMulÑ
-sequential/outputLayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential/outputLayer/BiasAdd/ReadVariableOpÝ
sequential/outputLayer/BiasAddBiasAdd'sequential/outputLayer/MatMul:product:05sequential/outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential/outputLayer/BiasAdd¦
sequential/outputLayer/SoftmaxSoftmax'sequential/outputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential/outputLayer/Softmax
IdentityIdentity(sequential/outputLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityï
NoOpNoOp/^sequential/hiddenLayer1/BiasAdd/ReadVariableOp.^sequential/hiddenLayer1/MatMul/ReadVariableOp/^sequential/hiddenLayer2/BiasAdd/ReadVariableOp.^sequential/hiddenLayer2/MatMul/ReadVariableOp.^sequential/outputLayer/BiasAdd/ReadVariableOp-^sequential/outputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.sequential/hiddenLayer1/BiasAdd/ReadVariableOp.sequential/hiddenLayer1/BiasAdd/ReadVariableOp2^
-sequential/hiddenLayer1/MatMul/ReadVariableOp-sequential/hiddenLayer1/MatMul/ReadVariableOp2`
.sequential/hiddenLayer2/BiasAdd/ReadVariableOp.sequential/hiddenLayer2/BiasAdd/ReadVariableOp2^
-sequential/hiddenLayer2/MatMul/ReadVariableOp-sequential/hiddenLayer2/MatMul/ReadVariableOp2^
-sequential/outputLayer/BiasAdd/ReadVariableOp-sequential/outputLayer/BiasAdd/ReadVariableOp2\
,sequential/outputLayer/MatMul/ReadVariableOp,sequential/outputLayer/MatMul/ReadVariableOp:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
õ

+__inference_outputLayer_layer_call_fn_39914

inputs
unknown:d

	unknown_0:

identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_395632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

ù
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39885

inputs1
matmul_readvariableop_resource:	¬d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
$
¶
E__inference_sequential_layer_call_and_return_conditional_losses_39782

inputs?
+hiddenlayer1_matmul_readvariableop_resource:
¬;
,hiddenlayer1_biasadd_readvariableop_resource:	¬>
+hiddenlayer2_matmul_readvariableop_resource:	¬d:
,hiddenlayer2_biasadd_readvariableop_resource:d<
*outputlayer_matmul_readvariableop_resource:d
9
+outputlayer_biasadd_readvariableop_resource:

identity¢#hiddenLayer1/BiasAdd/ReadVariableOp¢"hiddenLayer1/MatMul/ReadVariableOp¢#hiddenLayer2/BiasAdd/ReadVariableOp¢"hiddenLayer2/MatMul/ReadVariableOp¢"outputLayer/BiasAdd/ReadVariableOp¢!outputLayer/MatMul/ReadVariableOpu
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
inputLayer/Const
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inputLayer/Reshape¶
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp°
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/MatMul´
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOp¶
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/BiasAdd
hiddenLayer1/ReluReluhiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/Reluµ
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp³
hiddenLayer2/MatMulMatMulhiddenLayer1/Relu:activations:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/MatMul³
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOpµ
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/BiasAdd
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/Relu±
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp°
outputLayer/MatMulMatMulhiddenLayer2/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/MatMul°
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp±
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/BiasAdd
outputLayer/SoftmaxSoftmaxoutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/Softmaxx
IdentityIdentityoutputLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity­
NoOpNoOp$^hiddenLayer1/BiasAdd/ReadVariableOp#^hiddenLayer1/MatMul/ReadVariableOp$^hiddenLayer2/BiasAdd/ReadVariableOp#^hiddenLayer2/MatMul/ReadVariableOp#^outputLayer/BiasAdd/ReadVariableOp"^outputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2J
#hiddenLayer1/BiasAdd/ReadVariableOp#hiddenLayer1/BiasAdd/ReadVariableOp2H
"hiddenLayer1/MatMul/ReadVariableOp"hiddenLayer1/MatMul/ReadVariableOp2J
#hiddenLayer2/BiasAdd/ReadVariableOp#hiddenLayer2/BiasAdd/ReadVariableOp2H
"hiddenLayer2/MatMul/ReadVariableOp"hiddenLayer2/MatMul/ReadVariableOp2H
"outputLayer/BiasAdd/ReadVariableOp"outputLayer/BiasAdd/ReadVariableOp2F
!outputLayer/MatMul/ReadVariableOp!outputLayer/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å	

*__inference_sequential_layer_call_fn_39692
inputlayer_input
unknown:
¬
	unknown_0:	¬
	unknown_1:	¬d
	unknown_2:d
	unknown_3:d

	unknown_4:

identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_396602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameinputLayer_input
§	

*__inference_sequential_layer_call_fn_39843

inputs
unknown:
¬
	unknown_0:	¬
	unknown_1:	¬d
	unknown_2:d
	unknown_3:d

	unknown_4:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_396602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
¶
E__inference_sequential_layer_call_and_return_conditional_losses_39809

inputs?
+hiddenlayer1_matmul_readvariableop_resource:
¬;
,hiddenlayer1_biasadd_readvariableop_resource:	¬>
+hiddenlayer2_matmul_readvariableop_resource:	¬d:
,hiddenlayer2_biasadd_readvariableop_resource:d<
*outputlayer_matmul_readvariableop_resource:d
9
+outputlayer_biasadd_readvariableop_resource:

identity¢#hiddenLayer1/BiasAdd/ReadVariableOp¢"hiddenLayer1/MatMul/ReadVariableOp¢#hiddenLayer2/BiasAdd/ReadVariableOp¢"hiddenLayer2/MatMul/ReadVariableOp¢"outputLayer/BiasAdd/ReadVariableOp¢!outputLayer/MatMul/ReadVariableOpu
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
inputLayer/Const
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inputLayer/Reshape¶
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp°
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/MatMul´
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOp¶
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/BiasAdd
hiddenLayer1/ReluReluhiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
hiddenLayer1/Reluµ
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp³
hiddenLayer2/MatMulMatMulhiddenLayer1/Relu:activations:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/MatMul³
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOpµ
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/BiasAdd
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
hiddenLayer2/Relu±
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp°
outputLayer/MatMulMatMulhiddenLayer2/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/MatMul°
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp±
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/BiasAdd
outputLayer/SoftmaxSoftmaxoutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
outputLayer/Softmaxx
IdentityIdentityoutputLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity­
NoOpNoOp$^hiddenLayer1/BiasAdd/ReadVariableOp#^hiddenLayer1/MatMul/ReadVariableOp$^hiddenLayer2/BiasAdd/ReadVariableOp#^hiddenLayer2/MatMul/ReadVariableOp#^outputLayer/BiasAdd/ReadVariableOp"^outputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2J
#hiddenLayer1/BiasAdd/ReadVariableOp#hiddenLayer1/BiasAdd/ReadVariableOp2H
"hiddenLayer1/MatMul/ReadVariableOp"hiddenLayer1/MatMul/ReadVariableOp2J
#hiddenLayer2/BiasAdd/ReadVariableOp#hiddenLayer2/BiasAdd/ReadVariableOp2H
"hiddenLayer2/MatMul/ReadVariableOp"hiddenLayer2/MatMul/ReadVariableOp2H
"outputLayer/BiasAdd/ReadVariableOp"outputLayer/BiasAdd/ReadVariableOp2F
!outputLayer/MatMul/ReadVariableOp!outputLayer/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
F
*__inference_inputLayer_layer_call_fn_39854

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_395162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_39516

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
Q
inputLayer_input=
"serving_default_inputLayer_input:0ÿÿÿÿÿÿÿÿÿ?
outputLayer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:¥W
è
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
I_default_save_signature
*J&call_and_return_all_conditional_losses
K__call__"
_tf_keras_sequential
¥
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"
_tf_keras_layer
»

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*N&call_and_return_all_conditional_losses
O__call__"
_tf_keras_layer
»

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"
_tf_keras_layer
»

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
*R&call_and_return_all_conditional_losses
S__call__"
_tf_keras_layer
I
!iter
	"decay
#learning_rate
$momentum"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
Ê
regularization_losses
trainable_variables

%layers
&metrics
	variables
'non_trainable_variables
(layer_metrics
)layer_regularization_losses
K__call__
I_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,
Tserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
trainable_variables

*layers
+metrics
	variables
,non_trainable_variables
-layer_metrics
.layer_regularization_losses
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
':%
¬2hiddenLayer1/kernel
 :¬2hiddenLayer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
trainable_variables

/layers
0metrics
	variables
1non_trainable_variables
2layer_metrics
3layer_regularization_losses
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
&:$	¬d2hiddenLayer2/kernel
:d2hiddenLayer2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
trainable_variables

4layers
5metrics
	variables
6non_trainable_variables
7layer_metrics
8layer_regularization_losses
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
$:"d
2outputLayer/kernel
:
2outputLayer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
trainable_variables

9layers
:metrics
	variables
;non_trainable_variables
<layer_metrics
=layer_regularization_losses
S__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
<
0
1
2
3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	@total
	Acount
B	variables
C	keras_api"
_tf_keras_metric
^
	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
@0
A1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
ÔBÑ
 __inference__wrapped_model_39503inputLayer_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_39782
E__inference_sequential_layer_call_and_return_conditional_losses_39809
E__inference_sequential_layer_call_and_return_conditional_losses_39712
E__inference_sequential_layer_call_and_return_conditional_losses_39732À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
*__inference_sequential_layer_call_fn_39585
*__inference_sequential_layer_call_fn_39826
*__inference_sequential_layer_call_fn_39843
*__inference_sequential_layer_call_fn_39692À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_inputLayer_layer_call_and_return_conditional_losses_39849¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_inputLayer_layer_call_fn_39854¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39865¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_hiddenLayer1_layer_call_fn_39874¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39885¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_hiddenLayer2_layer_call_fn_39894¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_outputLayer_layer_call_and_return_conditional_losses_39905¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_outputLayer_layer_call_fn_39914¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÓBÐ
#__inference_signature_wrapper_39755inputLayer_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 §
 __inference__wrapped_model_39503=¢:
3¢0
.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
outputLayer%"
outputLayerÿÿÿÿÿÿÿÿÿ
©
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39865^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 
,__inference_hiddenLayer1_layer_call_fn_39874Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬¨
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39885]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
,__inference_hiddenLayer2_layer_call_fn_39894P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿd¦
E__inference_inputLayer_layer_call_and_return_conditional_losses_39849]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_inputLayer_layer_call_fn_39854P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_outputLayer_layer_call_and_return_conditional_losses_39905\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ~
+__inference_outputLayer_layer_call_fn_39914O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ
¿
E__inference_sequential_layer_call_and_return_conditional_losses_39712vE¢B
;¢8
.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¿
E__inference_sequential_layer_call_and_return_conditional_losses_39732vE¢B
;¢8
.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
E__inference_sequential_layer_call_and_return_conditional_losses_39782l;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
E__inference_sequential_layer_call_and_return_conditional_losses_39809l;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_sequential_layer_call_fn_39585iE¢B
;¢8
.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_39692iE¢B
;¢8
.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_39826_;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_39843_;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¾
#__inference_signature_wrapper_39755Q¢N
¢ 
GªD
B
inputLayer_input.+
inputLayer_inputÿÿÿÿÿÿÿÿÿ"9ª6
4
outputLayer%"
outputLayerÿÿÿÿÿÿÿÿÿ
