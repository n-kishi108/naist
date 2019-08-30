package main

import (
     "fmt"
     "github.com/PuerkitoBio/goquery"
)

func GetPage(url string) {
     doc, _ := goquery.NewDocument(url)
     doc.Find("img").Each(func(_ int, s *goquery.Selection) {
          url, _ := s.Attr("src")
          fmt.Println(url)
     })
}

func main() {
     var url string
     fmt.Println("URLを入力:")
     fmt.Scan(&url)
     //url := "http://blog.golang.org/"
     GetPage(url)
}

