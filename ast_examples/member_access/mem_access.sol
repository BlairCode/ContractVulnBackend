pragma solidity ^0.4.19;
contract MemberAccess {
	address constant owner = 0x00000000000000000000000000000000000000000000000000;
    function deposit() public payable {
		if (msg.sender == owner) {}
    }
}
