
main() {
    LayerInfo layers = ParseInputFiles();

    net.forward_read(layers);
}

void forward_read(layers)
{
    Mat input;
    Mat output;

    for (layer : layers) {
       blob = malloc(layer.size);
       output = computeLayer(input);
       input = output;
       free(blob);
    }

    return output;
}



//line 497
/** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputName name for layer which output is needed to get
         *  @return blob for first output of specified layer.
         *  @details By default runs forward pass for the whole network.
         */
        CV_WRAP Mat forward(const String& outputName = String());
