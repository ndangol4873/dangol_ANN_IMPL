"?L
BHostIDLE"IDLE1?????u?@A?????u?@a0???g???i0???g????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????̈?@9????̈?@A????̈?@I????̈?@aD????i?MS???Unknown?
?HostMatMul",gradient_tape/sequential/hiddenLayer1/MatMul(1?????\s@9?????\s@A?????\s@I?????\s@avp?Ϲ??ifh/Kj???Unknown
vHost_FusedMatMul"sequential/hiddenLayer1/Relu(133333s@933333s@A33333s@I33333s@a??#Rs??i??Lr?x???Unknown
?HostMatMul",gradient_tape/sequential/hiddenLayer2/MatMul(1ffffffR@9ffffffR@AffffffR@IffffffR@a??Ů?ɏ?i??%?????Unknown
vHost_FusedMatMul"sequential/hiddenLayer2/Relu(1     @P@9     @P@A     @P@I     @P@av?????i8=f?)h???Unknown
?HostMatMul".gradient_tape/sequential/hiddenLayer2/MatMul_1(1??????O@9??????O@A??????O@I??????O@at-I???i?zc6????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(133333sJ@933333sJ@A33333sJ@I33333sJ@a?O?"ن?iHl???1???Unknown
?	HostReadVariableOp"-sequential/hiddenLayer1/MatMul/ReadVariableOp(1fffff?F@9fffff?F@Afffff?F@Ifffff?F@a?{?͐??i\W#????Unknown
^
HostGatherV2"GatherV2(1fffff?A@9fffff?A@Afffff?A@Ifffff?A@a?57~~?i |ڼ???Unknown
sHostSoftmax"sequential/outputLayer/Softmax(1??????@@9??????@@A??????@@I??????@@a=?ҭ|?i??@76????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1??????;@9??????;@A??????;@I??????;@a??׫x?i̢??=&???Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?>@9     ?>@A??????;@I??????;@a??q?w?i???q?U???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1?????L;@9?????L;@A?????L;@I?????L;@aYp?w?i???????Unknown
HostMatMul"+gradient_tape/sequential/outputLayer/MatMul(1fffff?5@9fffff?5@Afffff?5@Ifffff?5@a1????r?i?m8??????Unknown
cHostDataset"Iterator::Root(1?????YA@9?????YA@A      5@I      5@a?????#r?ixUze4????Unknown
gHostStridedSlice"strided_slice(133333?2@933333?2@A33333?2@I33333?2@a?e?M'p?iCd&?????Unknown
iHostWriteSummary"WriteSummary(1      1@9      1@A      1@I      1@aH???^m?i?+7?????Unknown?
?HostMatMul"-gradient_tape/sequential/outputLayer/MatMul_1(1     ?0@9     ?0@A     ?0@I     ?0@a??j??l?i?ϡUc)???Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      .@9      .@A      .@I      .@am?7x-?i?i4?MC???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333-@9333333-@A333333-@I333333-@aAH?&D9i?i|?@ǆ\???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1ffffff+@9ffffff+@Affffff+@Iffffff+@a???.7?g?ic/o?1t???Unknown
?HostReluGrad".gradient_tape/sequential/hiddenLayer1/ReluGrad(1ffffff+@9ffffff+@Affffff+@Iffffff+@a???.7?g?iJ֝5݋???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff&@9ffffff&@Affffff&@Iffffff&@a?????Yc?i?ɍ?6????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1??????%@9??????%@A??????%@I??????%@a?)????b?iis??????Unknown
xHost_FusedMatMul"sequential/outputLayer/BiasAdd(1??????#@9??????#@A??????#@I??????#@a3??QT?`?i?g???????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_3(1      "@9      "@A      "@I      "@a??]_?i???f?????Unknown
`HostGatherV2"
GatherV2_1(1333333!@9333333!@A333333!@I333333!@a^Ӣ?0?]?i7???a????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @aۦ?t?[?i
@94????Unknown
?HostBiasAddGrad"9gradient_tape/sequential/hiddenLayer2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a?t? L[?iD?M9?????Unknown
?HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a????Z?iLʲ?'
???Unknown
? HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???ߞuV?i(?"?b???Unknown
t!HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a|??6*V?ik>)q ???Unknown
?"HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      +@9      +@A333333@I333333@a?X?An
T?i?^`v*???Unknown
?#HostBiasAddGrad"9gradient_tape/sequential/hiddenLayer1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??GS?i?d???3???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a?^a?&PR?i???=???Unknown
l%HostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a^???=?Q?i$?Κ?E???Unknown
?&HostBiasAddGrad"8gradient_tape/sequential/outputLayer/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aI????FQ?iSL??N???Unknown
Z'HostArgMax"ArgMax(1??????@9??????@A??????@I??????@a3??QT?P?iSMu)	W???Unknown
w(HostDataset""Iterator::Root::ParallelMapV2::Zip(1fffff?I@9fffff?I@A333333@I333333@ad?ߕP?i?IT_???Unknown
?)HostReluGrad".gradient_tape/sequential/hiddenLayer2/ReluGrad(1333333@9333333@A333333@I333333@ad?ߕP?i?f	?g???Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a??Ů??O?i7J??o???Unknown
?+HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1333333@9333333@A333333@I333333@a^Ӣ?0?M?i??xP?v???Unknown
e,Host
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a3o?gGM?i|R?@~???Unknown?
?-HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1333333@9333333@A333333@I333333@a?B?r??J?iY./??????Unknown
v.HostCast"$sparse_categorical_crossentropy/Cast(1ffffff@9ffffff@Affffff@Iffffff@a??!?BJ?is?m?????Unknown
?/HostReadVariableOp".sequential/hiddenLayer1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a+?}??H?i?ܖ?Ƒ???Unknown
?0HostReadVariableOp".sequential/hiddenLayer2/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a+?}??H?iFv??????Unknown
V1HostSum"Sum_2(1      @9      @A      @I      @a???+?/H?i	B?
????Unknown
a2HostIdentity"Identity(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?邈?F?i?b?S?????Unknown?
|3HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1??????@9??????@A??????@I??????@aP!?@lE?iK?d????Unknown
?4HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1??????@9??????@A??????@I??????@aP!?@lE?i??Utt????Unknown
?5HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a?X?An
D?i)X?w????Unknown
?6HostReadVariableOp"-sequential/hiddenLayer2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?????YC?i?T"qM????Unknown
X7HostEqual"Equal(1??????@9??????@A??????@I??????@au,?L??A?iqv?]˼???Unknown
?8HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      @I      @aI????FA?ic*?????Unknown
?9HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1333333@9333333@A333333@I333333@ad?ߕ@?i?pއB????Unknown
b:HostDivNoNan"div_no_nan_1(1ffffff@9ffffff@Affffff@Iffffff@a??Ů????i|It?;????Unknown
V;HostCast"Cast(1??????@9??????@A??????@I??????@a?7Yh>?i????????Unknown
X<HostCast"Cast_3(1??????@9??????@A??????@I??????@a?7Yh>?i????????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1?????? @9?????? @A?????? @I?????? @a3o?gG=?iX??v????Unknown
T>HostMul"Mul(1?????? @9?????? @A?????? @I?????? @a3o?gG=?i??]????Unknown
??HostReadVariableOp",sequential/outputLayer/MatMul/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a3o?gG=?it?&?????Unknown
?@HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @aۦ?t?;?ii?V?,????Unknown
XAHostCast"Cast_2(1ffffff??9ffffff??Affffff??Iffffff??a??!?B:?i?ʚ	u????Unknown
sBHostReadVariableOp"SGD/Cast/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??!?B:?i!??]?????Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333??9333333??A333333??I333333??a?M9??~7?iK4z=?????Unknown
uDHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a$?_?W?4?iC?l?D????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a$?_?W?4?i;_?????Unknown
yFHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a$?_?W?4?i3xQ~s????Unknown
?GHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a$?_?W?4?i+?C?
????Unknown
?HHostReadVariableOp"-sequential/outputLayer/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a$?_?W?4?i#P6T?????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a?????Y3?i?NԄ????Unknown
XJHostCast"Cast_4(1????????9????????A????????I????????au,?L??1?iH?{L????Unknown
?KHostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333??9333333??A333333??I333333??ad?ߕ0?iu7_????Unknown
`LHostDivNoNan"
div_no_nan(1????????9????????A????????I????????a?7Yh.?i???E????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aۦ?t?+?i     ???Unknown*?L
uHostFlushSummaryWriter"FlushSummaryWriter(1????̈?@9????̈?@A????̈?@I????̈?@ak;j?;??ik;j?;???Unknown?
?HostMatMul",gradient_tape/sequential/hiddenLayer1/MatMul(1?????\s@9?????\s@A?????\s@I?????\s@a??s?/???i0p??u???Unknown
vHost_FusedMatMul"sequential/hiddenLayer1/Relu(133333s@933333s@A33333s@I33333s@a?|?*?S??ikw?qZ???Unknown
?HostMatMul",gradient_tape/sequential/hiddenLayer2/MatMul(1ffffffR@9ffffffR@AffffffR@IffffffR@a??$?????i]?oQf???Unknown
vHost_FusedMatMul"sequential/hiddenLayer2/Relu(1     @P@9     @P@A     @P@I     @P@aײ?q[???i?Z(K?R???Unknown
?HostMatMul".gradient_tape/sequential/hiddenLayer2/MatMul_1(1??????O@9??????O@A??????O@I??????O@a??F7????i?????:???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(133333sJ@933333sJ@A33333sJ@I33333sJ@a!?)??i]8+?D????Unknown
?HostReadVariableOp"-sequential/hiddenLayer1/MatMul/ReadVariableOp(1fffff?F@9fffff?F@Afffff?F@Ifffff?F@a?Q4?????i??\e$????Unknown
^	HostGatherV2"GatherV2(1fffff?A@9fffff?A@Afffff?A@Ifffff?A@a???JI??i????? ???Unknown
s
HostSoftmax"sequential/outputLayer/Softmax(1??????@@9??????@@A??????@@I??????@@a?/?:i5??iIޞTt????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1??????;@9??????;@A??????;@I??????;@a?j?,?K??i?cQ??????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?>@9     ?>@A??????;@I??????;@a?F7????iA?c???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1?????L;@9?????L;@A?????L;@I?????L;@a?׈?iR???r????Unknown
HostMatMul"+gradient_tape/sequential/outputLayer/MatMul(1fffff?5@9fffff?5@Afffff?5@Ifffff?5@a?K??A???i?:??'???Unknown
cHostDataset"Iterator::Root(1?????YA@9?????YA@A      5@I      5@a???????i+] h?b???Unknown
gHostStridedSlice"strided_slice(133333?2@933333?2@A33333?2@I33333?2@awD????iim?ߥ????Unknown
iHostWriteSummary"WriteSummary(1      1@9      1@A      1@I      1@a?{ ??~?i?d"d?????Unknown?
?HostMatMul"-gradient_tape/sequential/outputLayer/MatMul_1(1     ?0@9     ?0@A     ?0@I     ?0@a<i?~?i?6%
? ???Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      .@9      .@A      .@I      .@a??0?L{?i??m+W???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333-@9333333-@A333333-@I333333-@a?d????z?i???nN????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1ffffff+@9ffffff+@Affffff+@Iffffff+@a?"͹d?x?i?C^8+????Unknown
?HostReluGrad".gradient_tape/sequential/hiddenLayer1/ReluGrad(1ffffff+@9ffffff+@Affffff+@Iffffff+@a?"͹d?x?i=??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff&@9ffffff&@Affffff&@Iffffff&@a?o?at?i??t????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1??????%@9??????%@A??????%@I??????%@a?91r??s?i??`w@???Unknown
xHost_FusedMatMul"sequential/outputLayer/BiasAdd(1??????#@9??????#@A??????#@I??????#@ab?????q?i[c?d"d???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_3(1      "@9      "@A      "@I      "@aO?P/?`p?i?G?????Unknown
`HostGatherV2"
GatherV2_1(1333333!@9333333!@A333333!@I333333!@a?P??Lo?i?T??0????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @aqYV??m?i઼?N????Unknown
?HostBiasAddGrad"9gradient_tape/sequential/hiddenLayer2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a??o??l?i?,,?????Unknown
?HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a??ى^l?iu??????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?(?'I?g?i???6????Unknown
t HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a'??Kg?i??S	)???Unknown
?!HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      +@9      +@A333333@I333333@a4e?i??e%>???Unknown
?"HostBiasAddGrad"9gradient_tape/sequential/hiddenLayer1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a~]???d?i?B?)R???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@at???3Jc?iޤ&te???Unknown
l$HostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@ak?J?ڏb?i?? x???Unknown
?%HostBiasAddGrad"8gradient_tape/sequential/outputLayer/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??um?2b?ieV?6????Unknown
Z&HostArgMax"ArgMax(1??????@9??????@A??????@I??????@ab?????a?i?Q1????Unknown
w'HostDataset""Iterator::Root::ParallelMapV2::Zip(1fffff?I@9fffff?I@A333333@I333333@a?h͇Uxa?i.?؆?????Unknown
?(HostReluGrad".gradient_tape/sequential/hiddenLayer2/ReluGrad(1333333@9333333@A333333@I333333@a?h͇Uxa?i??`??????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a??$???`?iq?ٺ????Unknown
?*HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1333333@9333333@A333333@I333333@a?P??L_?itnLPa????Unknown
e+Host
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a?w????^?i0B#??????Unknown?
?,HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1333333@9333333@A333333@I333333@agʭ??c\?i??`?????Unknown
v-HostCast"$sparse_categorical_crossentropy/Cast(1ffffff@9ffffff@Affffff@Iffffff@a^;2?[?i????
???Unknown
?.HostReadVariableOp".sequential/hiddenLayer1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@aL?K?4Z?i???9????Unknown
?/HostReadVariableOp".sequential/hiddenLayer2/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@aL?K?4Z?i???y?$???Unknown
V0HostSum"Sum_2(1      @9      @A      @I      @aC?f'zY?i?ի??1???Unknown
a1HostIdentity"Identity(1ffffff
@9ffffff
@Affffff
@Iffffff
@a0p??uX?i?2yH?=???Unknown?
|2HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1??????@9??????@A??????@I??????@aRi?ÐV?iy?`??H???Unknown
?3HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1??????@9??????@A??????@I??????@aRi?ÐV?i"?H6T???Unknown
?4HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a4U?i<?J?^???Unknown
?5HostReadVariableOp"-sequential/hiddenLayer2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?o?aT?i????h???Unknown
X6HostEqual"Equal(1??????@9??????@A??????@I??????@a??S?R?iRo?ukr???Unknown
?7HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      @I      @a??um?2R?iN*?̄{???Unknown
?8HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1333333@9333333@A333333@I333333@a?h͇UxQ?i~?@????Unknown
b9HostDivNoNan"div_no_nan_1(1ffffff@9ffffff@Affffff@Iffffff@a??$???P?io#???????Unknown
V:HostCast"Cast(1??????@9??????@A??????@I??????@a?J|??P?i?a?ǡ????Unknown
X;HostCast"Cast_3(1??????@9??????@A??????@I??????@a?J|??P?i?????????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1?????? @9?????? @A?????? @I?????? @a?w????N?i?	?>H????Unknown
T=HostMul"Mul(1?????? @9?????? @A?????? @I?????? @a?w????N?iusb??????Unknown
?>HostReadVariableOp",sequential/outputLayer/MatMul/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a?w????N?iS?͉?????Unknown
??HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @aqYV??M?i?r?ٺ???Unknown
X@HostCast"Cast_2(1ffffff??9ffffff??Affffff??Iffffff??a^;2?K?i84LO?????Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a^;2?K?i??ћ?????Unknown
wBHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333??9333333??A333333??I333333??a9?b?οH?iGr??????Unknown
uCHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a???j?E?ix~,*S????Unknown
uDHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a???j?E?i?????????Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???j?E?i?^?_>????Unknown
?FHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a???j?E?i?[??????Unknown
?GHostReadVariableOp"-sequential/outputLayer/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a???j?E?i<??)????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a?o?aD?i%?]B????Unknown
XIHostCast"Cast_4(1????????9????????A????????I????????a??S?B?iǢ2E?????Unknown
?JHostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333??9333333??A333333??I333333??a?h͇UxA?i!??Z[????Unknown
`KHostDivNoNan"
div_no_nan(1????????9????????A????????I????????a?J|??@?i4??C\????Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aqYV??=?i?????????Unknown2CPU