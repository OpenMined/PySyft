def test_data_subjects(client, data_subjects):
    response = client.data_subject_registry.add_data_subject(data_subjects)

    assert data_subjects in client.data_subject_registry.get_all()
    assert response
