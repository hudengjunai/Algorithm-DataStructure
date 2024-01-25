//list unorderd,has repeate nums 
//1.convert vector to list 
//2.sort list 
//3.remove list dup element 


#include<iostream>
#include<vector>
using namespace std;
struct ListNode{
    int val;
    ListNode* next;
    ListNode():val(0),next(nullptr){}
};

ListNode* convert_vector_list(const vector<int>& nums){
    ListNode* dummy = new ListNode();
    ListNode* prev = dummy;
    for(const auto v: nums){
        ListNode* n_node = new ListNode();
        n_node->val = v;
        prev->next = n_node;
        prev = prev->next;
    }
    ListNode* head = dummy->next;
    delete dummy;
    return head;
}

void print_list(ListNode* list){
    ListNode* cur = list;
    while(cur){
        cout<<cur->val<<",";
        cur = cur->next;
    }
    cout<<endl;
}
//asuume l1, l2 both sorted.
ListNode* mergeList(ListNode* l1, ListNode* l2){
    ListNode* dummy = new ListNode();
    ListNode* prev = dummy;
    ListNode* l1_c = l1;
    ListNode* l2_c = l2;
    while(l1_c && l2_c){
        if(l1_c->val < l2_c->val){
            prev->next = l1_c;
            l1_c = l1_c->next;
        }else{
            prev->next = l2_c;
            l2_c = l2_c->next;
        }
        prev = prev->next;
    }
    if(l1_c){
        prev->next = l1_c;
    }
    if(l2_c){
        prev->next = l2_c;
    }
    ListNode* head = dummy->next;
    delete dummy;
    return head;
}
ListNode* sortList(ListNode* head, ListNode* tail){
    if(head == tail){
        return head;
    }
    if(head->next==tail){
        head->next=nullptr;
        return head;
    }
    //split the head->list into two list l1,l2
    ListNode* slow = head;
    ListNode* fast = head;
    while(fast !=tail ){
        slow =slow->next;
        fast = fast->next;
        if(fast){
            fast = fast->next;
        }
    }
    ListNode* mid = slow;
    
    ListNode* l1 = sortList(head,mid);
    ListNode* l2 = sortList(mid,tail);
    //print_list(l1);
    //print_list(l2);
    return mergeList(l1,l2);
}



ListNode* remove_dup_ele(ListNode* l){
    //remove 
    ListNode* prev =nullptr;
    ListNode* cur = l;
    while(cur){
        //cout<<"cur val"<<cur->val;
        if(prev && cur->val==prev->val){
            //ListNode* d_node = cur;
            //delete d_node;
            prev->next = cur->next;
        }
        prev = cur;
        cur = cur->next;
    }
    return l;
}
int main(){
    vector<int> nums{3,2,2,5,1,19,5};
    ListNode* head = convert_vector_list(nums);
    print_list(head);
    ListNode* sorted_list = sortList(head,nullptr);
    print_list(sorted_list);
    cout<<"sorted 2 "<<endl;
    print_list(sorted_list);
    auto rd_l = remove_dup_ele(sorted_list);
    print_list(rd_l);
    return 0;
}
