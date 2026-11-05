from lesson_harness_v1.word_transcripts import pack_contained_words


def test_word_boundary_exclusion_and_no_sentence_leakage():
    spans=[{'id':str(i),'start':a,'stop':b,'text':t,'sentence':'FUTURE SECRET'}
           for i,(a,b,t) in enumerate([(0,2,'before'),(2,3,'in'),(40,41,'context'),(43,46,'future')])]
    result=pack_contained_words(spans,1,45)
    assert result[0]['text']=='in context'
    assert result[0]['word_ids']==['1','2']
    assert result[0]['start']==2 and result[0]['stop']==41
    assert 'FUTURE' not in str(result)
    assert pack_contained_words(spans,50,60)==[]
