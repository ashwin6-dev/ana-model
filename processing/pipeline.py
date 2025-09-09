class Processor:
    def execute(self, input_data):
        pass

class Pipeline:
    def __init__(self, processors):
        self.processors = processors

    def run(self, pipeline_input):
        result = pipeline_input
        for processor in self.processors:
            result = processor.execute(result)
        return result
