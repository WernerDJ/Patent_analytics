from django.test import SimpleTestCase
from django.urls import reverse, resolve
from .views import HomePageView

class HomepageTests(SimpleTestCase):
    def setUp(self): 
        url = reverse("home")
        self.response = self.client.get(url)
    '''
    test that the homepage exists, but we should also confirm that it uses the correct template
    '''
    def test_url_exists_at_correct_location(self):
        self.assertEqual(self.response.status_code, 200)
    def test_homepage_template(self):
        self.assertTemplateUsed(self.response, "home.html")
    '''
    confirm that homepage has the correct HTML code and also does not have incorrect text
    '''
    def test_homepage_contains_correct_html(self):
        self.assertContains(self.response, "Patent Analysis")
    def test_homepage_does_not_contain_incorrect_html(self):
        self.assertNotContains(self.response, "Hi there! I should not be on the page.")
    '''
    A final views check that the  HomePageView “resolves” a given URL path. Django
    contains the utility function resolve for just this purpose. We will need to import both resolve
    as well as the HomePageView at the top of the file.
    '''
    def test_homepage_url_resolves_homepageview(self): 
        view = resolve("/")
        self.assertEqual(view.func.__name__, HomePageView.as_view().__name__)
