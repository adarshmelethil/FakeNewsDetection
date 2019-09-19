FROM ubuntu:18.04
COPY . /FakeNewsDetection
WORKDIR /FakeNewsDetection

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    python3-pip \
    python3-setuptools

RUN python3 setup.py install
RUN python3 -m spacy download en_core_web_sm

#Replace with appropriate command
CMD DataCup_test_search
