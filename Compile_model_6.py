class Compilemodel():
    def compilemodel(self,model_):
        model_.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

        model_.summary()
        return model_
